// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! DataFusion physical expression and projection for computing embeddings.

use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::ExecutionPlan;

use crate::embeddings::{EmbeddingDefinition, EmbeddingFunction};

/// Wraps an [`EmbeddingFunction`] as a DataFusion physical expression.
#[derive(Debug)]
pub struct EmbeddingPhysicalExpr {
    source_expr: Arc<dyn PhysicalExpr>,
    embedding_function: Arc<dyn EmbeddingFunction>,
    return_type: DataType,
}

impl PartialEq for EmbeddingPhysicalExpr {
    fn eq(&self, other: &Self) -> bool {
        self.source_expr.eq(&other.source_expr)
            && Arc::ptr_eq(&self.embedding_function, &other.embedding_function)
            && self.return_type == other.return_type
    }
}

impl Eq for EmbeddingPhysicalExpr {}

impl Hash for EmbeddingPhysicalExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source_expr.hash(state);
        (Arc::as_ptr(&self.embedding_function) as *const () as usize).hash(state);
        self.return_type.hash(state);
    }
}

impl fmt::Display for EmbeddingPhysicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embedding({}, func={})",
            self.source_expr,
            self.embedding_function.name()
        )
    }
}

impl EmbeddingPhysicalExpr {
    pub fn new(
        source_expr: Arc<dyn PhysicalExpr>,
        embedding_function: Arc<dyn EmbeddingFunction>,
    ) -> crate::Result<Self> {
        let raw_type = embedding_function.dest_type()?.into_owned();
        // Normalize FSL inner field to nullable to match DataFusion schema conventions
        let return_type = match raw_type {
            DataType::FixedSizeList(inner, dim) => DataType::FixedSizeList(
                Arc::new(Field::new(inner.name(), inner.data_type().clone(), true)),
                dim,
            ),
            other => other,
        };
        Ok(Self {
            source_expr,
            embedding_function,
            return_type,
        })
    }
}

impl PhysicalExpr for EmbeddingPhysicalExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> DFResult<DataType> {
        Ok(self.return_type.clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> DFResult<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &arrow_array::RecordBatch) -> DFResult<ColumnarValue> {
        let source_value = self.source_expr.evaluate(batch)?;
        let array: ArrayRef = source_value.into_array(batch.num_rows())?;
        let result = self
            .embedding_function
            .compute_source_embeddings(array)
            .map_err(|e| DataFusionError::External(e.into()))?;
        // Ensure the result array matches the schema type from data_type().
        // DataFusion's ProjectionExec may normalize inner field nullability,
        // so we cast if needed to avoid schema mismatches.
        let result = arrow_cast::cast(&result, &self.return_type)?;
        Ok(ColumnarValue::Array(result))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.source_expr]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self {
            source_expr: children[0].clone(),
            embedding_function: self.embedding_function.clone(),
            return_type: self.return_type.clone(),
        }))
    }

    fn fmt_sql(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

/// Builds a [`ProjectionExec`] that appends embedding columns to the input.
///
/// For each embedding definition, creates an [`EmbeddingPhysicalExpr`] that
/// computes the embedding from the source column and adds it as a new column.
/// If the destination column already exists in the input schema, it is skipped.
///
/// Returns the input unchanged if `embeddings` is empty or no new columns are needed.
pub fn build_embedding_projection(
    input: Arc<dyn ExecutionPlan>,
    embeddings: &[(EmbeddingDefinition, Arc<dyn EmbeddingFunction>)],
) -> DFResult<Arc<dyn ExecutionPlan>> {
    if embeddings.is_empty() {
        return Ok(input);
    }

    let input_schema = input.schema();

    // Start with passthrough for all existing columns
    let mut projections: Vec<(Arc<dyn PhysicalExpr>, String)> = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(f.name(), i));
            (expr, f.name().clone())
        })
        .collect();

    let mut added = false;
    for (def, func) in embeddings {
        let dest_name = def
            .dest_column
            .clone()
            .unwrap_or_else(|| format!("{}_embedding", def.source_column));

        if input_schema.field_with_name(&dest_name).is_ok() {
            continue;
        }

        let source_idx = input_schema.index_of(&def.source_column)?;
        let source_expr: Arc<dyn PhysicalExpr> =
            Arc::new(Column::new(&def.source_column, source_idx));
        let embed_expr = EmbeddingPhysicalExpr::new(source_expr, func.clone())
            .map_err(|e| DataFusionError::External(e.into()))?;
        projections.push((Arc::new(embed_expr), dest_name));
        added = true;
    }

    if !added {
        return Ok(input);
    }

    Ok(Arc::new(ProjectionExec::try_new(projections, input)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_catalog::MemTable;
    use datafusion_physical_plan::collect;

    use crate::test_utils::embeddings::MockEmbed;

    fn make_text_batch() -> (Arc<Schema>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["hello", "world"]))],
        )
        .unwrap();
        (schema, batch)
    }

    #[test]
    fn test_embedding_expr_evaluate() {
        let (_schema, batch) = make_text_batch();
        let mock = Arc::new(MockEmbed::new("test", 4));
        let source_expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("text", 0));
        let expr = EmbeddingPhysicalExpr::new(source_expr, mock).unwrap();

        let result = expr.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(arr) => {
                let fsl = arr.as_fixed_size_list();
                assert_eq!(fsl.len(), 2);
                assert_eq!(fsl.value_length(), 4);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_embedding_expr_data_type() {
        let mock = Arc::new(MockEmbed::new("test", 4));
        let source_expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("text", 0));
        let expr = EmbeddingPhysicalExpr::new(source_expr, mock).unwrap();

        let schema = Schema::new(vec![Field::new("text", DataType::Utf8, false)]);
        let dt = expr.data_type(&schema).unwrap();
        assert_eq!(
            dt,
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4)
        );
    }

    #[tokio::test]
    async fn test_build_projection_adds_column() {
        let (schema, batch) = make_text_batch();
        let mock: Arc<dyn EmbeddingFunction> = Arc::new(MockEmbed::new("test", 4));
        let def = EmbeddingDefinition::new("text", "test", Some("text_vec"));

        let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem_table)).unwrap();
        let df = ctx.sql("SELECT * FROM t").await.unwrap();
        let input = df.create_physical_plan().await.unwrap();

        let result_plan = build_embedding_projection(input, &[(def, mock)]).unwrap();
        let batches = collect(result_plan, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.len(), 1);
        let result = &batches[0];
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(1).name(), "text_vec");
        let fsl = result.column(1).as_fixed_size_list();
        assert_eq!(fsl.len(), 2);
        assert_eq!(fsl.value_length(), 4);
    }

    #[tokio::test]
    async fn test_build_projection_skips_existing() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("text_vec", DataType::Float32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["hello", "world"])),
                Arc::new(arrow_array::Float32Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();

        let mock: Arc<dyn EmbeddingFunction> = Arc::new(MockEmbed::new("test", 4));
        let def = EmbeddingDefinition::new("text", "test", Some("text_vec"));

        let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem_table)).unwrap();
        let df = ctx.sql("SELECT * FROM t").await.unwrap();
        let input = df.create_physical_plan().await.unwrap();

        let result_plan = build_embedding_projection(input.clone(), &[(def, mock)]).unwrap();
        // Should return input unchanged since dest column already exists
        assert!(Arc::ptr_eq(&result_plan, &input));
    }

    #[tokio::test]
    async fn test_build_projection_default_name() {
        let (schema, batch) = make_text_batch();
        let mock: Arc<dyn EmbeddingFunction> = Arc::new(MockEmbed::new("test", 4));
        // No dest_column → defaults to "{source}_embedding"
        let def = EmbeddingDefinition::new("text", "test", None::<&str>);

        let mem_table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem_table)).unwrap();
        let df = ctx.sql("SELECT * FROM t").await.unwrap();
        let input = df.create_physical_plan().await.unwrap();

        let result_plan = build_embedding_projection(input, &[(def, mock)]).unwrap();
        let batches = collect(result_plan, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].schema().field(1).name(), "text_embedding");
    }
}

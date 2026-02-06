// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! DataFusion ExecutionPlan node for NaN handling in float vector columns.

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result as DataFusionResult};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::TryStreamExt;

use crate::data::preprocessing::{find_float_vector_columns, handle_nan_batch, NanStrategy};

/// Streaming plan node that handles NaN values in float vector columns.
///
/// Applies a [`NanStrategy`] to every batch, processing all float vector
/// columns (`FixedSizeList`, `List`, `LargeList` with float elements).
#[derive(Debug)]
pub struct PreprocessingExec {
    input: Arc<dyn ExecutionPlan>,
    strategy: NanStrategy,
    vec_col_indices: Vec<usize>,
    properties: PlanProperties,
}

impl PreprocessingExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, strategy: NanStrategy) -> Self {
        let schema = input.schema();
        let vec_col_indices = find_float_vector_columns(&schema);
        let properties = Self::compute_properties(&input, &schema);
        Self {
            input,
            strategy,
            vec_col_indices,
            properties,
        }
    }

    fn compute_properties(input: &Arc<dyn ExecutionPlan>, schema: &SchemaRef) -> PlanProperties {
        let input_props = input.properties();
        let eq_properties = EquivalenceProperties::new(schema.clone());
        PlanProperties::new(
            eq_properties,
            input_props.output_partitioning().clone(),
            match input_props.emission_type {
                EmissionType::Incremental => EmissionType::Incremental,
                EmissionType::Final => EmissionType::Final,
                EmissionType::Both => EmissionType::Both,
            },
            match input_props.boundedness {
                Boundedness::Bounded => Boundedness::Bounded,
                other => other,
            },
        )
    }
}

impl DisplayAs for PreprocessingExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PreprocessingExec: strategy={:?}, vec_cols={:?}",
                    self.strategy, self.vec_col_indices
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "PreprocessingExec")
            }
        }
    }
}

impl ExecutionPlan for PreprocessingExec {
    fn name(&self) -> &str {
        "PreprocessingExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "PreprocessingExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.strategy.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;

        if self.vec_col_indices.is_empty() {
            return Ok(input_stream);
        }

        let schema = self.input.schema();
        let strategy = self.strategy.clone();
        let vec_cols = self.vec_col_indices.clone();

        let stream = input_stream
            .map_ok(move |batch| handle_nan_batch(batch, &vec_cols, &strategy, &schema));

        // Flatten the Result<Result<RecordBatch>> into Result<RecordBatch>
        let stream = stream.and_then(|result| async move {
            result.map_err(|e| DataFusionError::External(e.into()))
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_catalog::MemTable;
    use datafusion_physical_plan::collect;

    fn make_fsl_f32(values: &[Option<Vec<f32>>], dim: i32) -> ArrayRef {
        let dim_usize = dim as usize;
        let n = values.len();
        let mut builder =
            FixedSizeListBuilder::new(Float32Builder::with_capacity(n * dim_usize), dim);
        for row in values {
            match row {
                Some(vals) => {
                    for v in vals {
                        builder.values().append_value(*v);
                    }
                    builder.append(true);
                }
                None => {
                    for _ in 0..dim_usize {
                        builder.values().append_null();
                    }
                    builder.append(false);
                }
            }
        }
        Arc::new(builder.finish())
    }

    fn fsl_schema(dim: i32) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]))
    }

    async fn run_preprocessing(
        schema: Arc<Schema>,
        batch: RecordBatch,
        strategy: NanStrategy,
    ) -> DataFusionResult<Vec<RecordBatch>> {
        let mem_table = MemTable::try_new(schema, vec![vec![batch]])?;
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem_table))?;
        let df = ctx.sql("SELECT * FROM t").await?;
        let input = df.create_physical_plan().await?;
        let preprocessing = Arc::new(PreprocessingExec::new(input, strategy));
        collect(preprocessing, ctx.task_ctx()).await
    }

    #[tokio::test]
    async fn test_error_strategy() {
        let schema = fsl_schema(3);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                make_fsl_f32(
                    &[Some(vec![1.0, 2.0, 3.0]), Some(vec![f32::NAN, 5.0, 6.0])],
                    3,
                ),
            ],
        )
        .unwrap();

        let result = run_preprocessing(schema, batch, NanStrategy::Error).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_drop_strategy() {
        let schema = fsl_schema(3);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                make_fsl_f32(
                    &[
                        Some(vec![1.0, 2.0, 3.0]),
                        Some(vec![f32::NAN, 5.0, 6.0]),
                        Some(vec![7.0, 8.0, 9.0]),
                    ],
                    3,
                ),
            ],
        )
        .unwrap();

        let batches = run_preprocessing(schema, batch, NanStrategy::Drop)
            .await
            .unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        let ids = batches[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(1), 3);
    }

    #[tokio::test]
    async fn test_null_strategy() {
        let schema = fsl_schema(3);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                make_fsl_f32(
                    &[
                        Some(vec![1.0, 2.0, 3.0]),
                        Some(vec![f32::NAN, 5.0, 6.0]),
                        Some(vec![7.0, 8.0, 9.0]),
                    ],
                    3,
                ),
            ],
        )
        .unwrap();

        let batches = run_preprocessing(schema, batch, NanStrategy::Null)
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let result = &batches[0];
        assert_eq!(result.num_rows(), 3);
        let fsl = result.column(1).as_fixed_size_list();
        assert!(!fsl.is_null(0));
        assert!(fsl.is_null(1)); // NaN row → null
        assert!(!fsl.is_null(2));
    }

    #[tokio::test]
    async fn test_fill_strategy() {
        let schema = fsl_schema(3);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                make_fsl_f32(
                    &[Some(vec![1.0, 2.0, 3.0]), Some(vec![f32::NAN, 5.0, 6.0])],
                    3,
                ),
            ],
        )
        .unwrap();

        let batches = run_preprocessing(schema, batch, NanStrategy::Fill(0.0))
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let result = &batches[0];
        assert_eq!(result.num_rows(), 2);
        let fsl = result.column(1).as_fixed_size_list();
        let row1 = fsl.value(1);
        let vals = row1.as_primitive::<Float32Type>();
        assert_eq!(vals.value(0), 0.0); // was NaN, now 0.0
        assert_eq!(vals.value(1), 5.0);
        assert_eq!(vals.value(2), 6.0);
    }

    #[tokio::test]
    async fn test_no_vector_cols_passthrough() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(arrow_array::StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();

        let batches = run_preprocessing(schema, batch, NanStrategy::Error)
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[tokio::test]
    async fn test_clean_data_passthrough() {
        let schema = fsl_schema(3);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                make_fsl_f32(&[Some(vec![1.0, 2.0, 3.0]), Some(vec![4.0, 5.0, 6.0])], 3),
            ],
        )
        .unwrap();

        let batches = run_preprocessing(schema, batch, NanStrategy::Error)
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
    }
}

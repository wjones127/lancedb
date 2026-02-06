// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! DataFusion physical expression and projection for casting vector columns to `FixedSizeList`.

use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result as DataFusionResult;

type DFResult<T> = DataFusionResult<T>;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::expressions::{CastExpr, Column};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::ExecutionPlan;

use crate::data::preprocessing::{
    build_fsl_from_offsets, get_list_offsets_and_values, nan_row_mask, nullify_fsl_rows,
    BadVectorStrategy,
};

/// Casts `List`/`LargeList`/`FixedSizeList` columns to `FixedSizeList<T, N>`.
#[derive(Debug, Clone, Eq)]
pub struct CastToFixedSizeListExpr {
    source: Arc<dyn PhysicalExpr>,
    dimension: i32,
    inner_type: DataType,
}

impl PartialEq for CastToFixedSizeListExpr {
    fn eq(&self, other: &Self) -> bool {
        self.source.eq(&other.source)
            && self.dimension == other.dimension
            && self.inner_type == other.inner_type
    }
}

impl Hash for CastToFixedSizeListExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.dimension.hash(state);
        self.inner_type.hash(state);
    }
}

impl fmt::Display for CastToFixedSizeListExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CastToFSL({}, dim={}, inner={})",
            self.source, self.dimension, self.inner_type
        )
    }
}

impl CastToFixedSizeListExpr {
    pub fn new(source: Arc<dyn PhysicalExpr>, dimension: i32, inner_type: DataType) -> Self {
        Self {
            source,
            dimension,
            inner_type,
        }
    }

    pub fn try_new_from_target_type(
        source: Arc<dyn PhysicalExpr>,
        target: &DataType,
    ) -> crate::Result<Self> {
        match target {
            DataType::FixedSizeList(inner, dim) => {
                Ok(Self::new(source, *dim, inner.data_type().clone()))
            }
            _ => Err(crate::Error::InvalidInput {
                message: format!("expected FixedSizeList target type, got {:?}", target),
            }),
        }
    }

    fn cast_fsl(&self, fsl: &FixedSizeListArray) -> datafusion_common::Result<ArrayRef> {
        if fsl.value_length() != self.dimension {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "FSL dimension mismatch: expected {}, got {}",
                self.dimension,
                fsl.value_length()
            )));
        }
        let values = fsl.values();
        if *values.data_type() == self.inner_type {
            return Ok(Arc::new(fsl.clone()));
        }
        let cast_values = cast(values, &self.inner_type)?;
        let field = Arc::new(Field::new("item", self.inner_type.clone(), true));
        let new_fsl =
            FixedSizeListArray::new(field, self.dimension, cast_values, fsl.nulls().cloned());
        Ok(Arc::new(new_fsl))
    }
}

impl PhysicalExpr for CastToFixedSizeListExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> DFResult<DataType> {
        Ok(DataType::FixedSizeList(
            Arc::new(Field::new("item", self.inner_type.clone(), true)),
            self.dimension,
        ))
    }

    fn nullable(&self, _input_schema: &Schema) -> DFResult<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &arrow_array::RecordBatch) -> DFResult<ColumnarValue> {
        let source_value = self.source.evaluate(batch)?;
        let array = source_value.into_array(batch.num_rows())?;

        let result = match array.data_type() {
            DataType::List(_) | DataType::LargeList(_) => {
                let (offsets, flat_values) = get_list_offsets_and_values(array.as_ref())
                    .map_err(|e| datafusion_common::DataFusionError::External(e.into()))?;
                let cast_values = cast(&flat_values, &self.inner_type)?;

                match &self.inner_type {
                    DataType::Float32 => {
                        let typed = cast_values.as_primitive::<arrow_array::types::Float32Type>();
                        build_fsl_from_offsets::<arrow_array::builder::Float32Builder>(
                            array.as_ref(),
                            &offsets,
                            typed,
                            self.dimension as i64,
                            &BadVectorStrategy::Null,
                        )
                        .map_err(|e| datafusion_common::DataFusionError::External(e.into()))?
                    }
                    DataType::UInt8 => {
                        let typed = cast_values
                            .as_any()
                            .downcast_ref::<arrow_array::UInt8Array>()
                            .expect("cast to UInt8 should succeed");
                        build_fsl_from_offsets::<arrow_array::builder::UInt8Builder>(
                            array.as_ref(),
                            &offsets,
                            typed,
                            self.dimension as i64,
                            &BadVectorStrategy::Null,
                        )
                        .map_err(|e| datafusion_common::DataFusionError::External(e.into()))?
                    }
                    _ => {
                        return Err(datafusion_common::DataFusionError::Execution(format!(
                            "unsupported inner type for FSL cast: {:?}",
                            self.inner_type
                        )));
                    }
                }
            }
            DataType::FixedSizeList(..) => self.cast_fsl(array.as_fixed_size_list())?,
            other => {
                return Err(datafusion_common::DataFusionError::Execution(format!(
                    "CastToFixedSizeListExpr: unsupported source type {:?}",
                    other
                )));
            }
        };

        // NaN defense-in-depth: nullify rows with NaN
        let fsl = result.as_fixed_size_list();
        let is_float = matches!(
            fsl.value_type(),
            DataType::Float16 | DataType::Float32 | DataType::Float64
        );
        let result = if is_float {
            let mask = nan_row_mask(fsl);
            let has_nan = mask.iter().any(|v| v == Some(true));
            if has_nan {
                nullify_fsl_rows(fsl, &mask)
                    .map_err(|e| datafusion_common::DataFusionError::External(e.into()))?
            } else {
                result
            }
        } else {
            result
        };

        Ok(ColumnarValue::Array(result))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.source]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.dimension,
            self.inner_type.clone(),
        )))
    }

    fn fmt_sql(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

fn schemas_equal(a: &Schema, b: &Schema) -> bool {
    if a.fields().len() != b.fields().len() {
        return false;
    }
    a.fields()
        .iter()
        .zip(b.fields().iter())
        .all(|(af, bf)| af.name() == bf.name() && af.data_type() == bf.data_type())
}

fn needs_fsl_conversion(source_type: &DataType, target_type: &DataType) -> bool {
    matches!(target_type, DataType::FixedSizeList(..))
        && matches!(
            source_type,
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..)
        )
        && source_type != target_type
}

/// Builds a [`ProjectionExec`] adapting input schema to target schema.
///
/// For each target field, finds the corresponding source field by name and
/// applies the appropriate cast: `CastToFixedSizeListExpr` for list-to-FSL
/// conversions, standard `CastExpr` for other type mismatches, or a simple
/// column passthrough if types already match.
///
/// Returns the input unchanged if schemas already match.
pub fn create_schema_cast_projection(
    input: Arc<dyn ExecutionPlan>,
    target_schema: &SchemaRef,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let input_schema = input.schema();
    if schemas_equal(&input_schema, target_schema) {
        return Ok(input);
    }

    let mut projections: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::new();

    for target_field in target_schema.fields() {
        let name = target_field.name();
        let source_idx = input_schema.index_of(name)?;
        let source_field = input_schema.field(source_idx);
        let col_expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(name, source_idx));

        if source_field.data_type() == target_field.data_type() {
            projections.push((col_expr, name.clone()));
        } else if needs_fsl_conversion(source_field.data_type(), target_field.data_type()) {
            let cast_expr = CastToFixedSizeListExpr::try_new_from_target_type(
                col_expr,
                target_field.data_type(),
            )
            .map_err(|e| datafusion_common::DataFusionError::External(e.into()))?;
            projections.push((Arc::new(cast_expr), name.clone()));
        } else {
            let cast_expr = CastExpr::new(col_expr, target_field.data_type().clone(), None);
            projections.push((Arc::new(cast_expr), name.clone()));
        }
    }

    Ok(Arc::new(ProjectionExec::try_new(projections, input)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{
        FixedSizeListBuilder, Float32Builder, Float64Builder, LargeListBuilder, ListBuilder,
    };
    use arrow_array::types::Float32Type;
    use arrow_array::{Float32Array, RecordBatch};
    use arrow_schema::Field;
    use datafusion::prelude::SessionContext;
    use datafusion_catalog::MemTable;
    use datafusion_physical_plan::collect;

    fn make_list_f32(values: &[Option<Vec<f32>>]) -> ArrayRef {
        let mut builder = ListBuilder::new(Float32Builder::new());
        for row in values {
            match row {
                Some(vals) => {
                    for v in vals {
                        builder.values().append_value(*v);
                    }
                    builder.append(true);
                }
                None => {
                    builder.append(false);
                }
            }
        }
        Arc::new(builder.finish())
    }

    fn make_list_f64(values: &[Option<Vec<f64>>]) -> ArrayRef {
        let mut builder = ListBuilder::new(Float64Builder::new());
        for row in values {
            match row {
                Some(vals) => {
                    for v in vals {
                        builder.values().append_value(*v);
                    }
                    builder.append(true);
                }
                None => {
                    builder.append(false);
                }
            }
        }
        Arc::new(builder.finish())
    }

    fn make_large_list_f32(values: &[Option<Vec<f32>>]) -> ArrayRef {
        let mut builder = LargeListBuilder::new(Float32Builder::new());
        for row in values {
            match row {
                Some(vals) => {
                    for v in vals {
                        builder.values().append_value(*v);
                    }
                    builder.append(true);
                }
                None => {
                    builder.append(false);
                }
            }
        }
        Arc::new(builder.finish())
    }

    fn make_fsl_f64(values: &[Option<Vec<f64>>], dim: i32) -> ArrayRef {
        let dim_usize = dim as usize;
        let n = values.len();
        let mut builder =
            FixedSizeListBuilder::new(Float64Builder::with_capacity(n * dim_usize), dim);
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

    fn eval_expr(expr: &CastToFixedSizeListExpr, batch: &RecordBatch) -> ArrayRef {
        match expr.evaluate(batch).unwrap() {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        }
    }

    fn source_col() -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new("vec", 0))
    }

    #[test]
    fn test_list_f32_to_fsl() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_list_f32(&[
                Some(vec![1.0, 2.0, 3.0]),
                Some(vec![4.0, 5.0, 6.0]),
            ])],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 3, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert_eq!(fsl.len(), 2);
        assert_eq!(fsl.value_length(), 3);
        let row0 = fsl.value(0);
        let vals = row0.as_primitive::<Float32Type>();
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(1), 2.0);
        assert_eq!(vals.value(2), 3.0);
    }

    #[test]
    fn test_list_f64_to_fsl_f32() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_list_f64(&[Some(vec![1.0, 2.0]), Some(vec![3.0, 4.0])])],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 2, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert_eq!(fsl.len(), 2);
        assert_eq!(fsl.value_type(), DataType::Float32);
        let row0 = fsl.value(0);
        let vals = row0.as_primitive::<Float32Type>();
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(1), 2.0);
    }

    #[test]
    fn test_large_list_to_fsl() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_large_list_f32(&[
                Some(vec![1.0, 2.0]),
                Some(vec![3.0, 4.0]),
            ])],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 2, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert_eq!(fsl.len(), 2);
        assert_eq!(fsl.value_length(), 2);
    }

    #[test]
    fn test_fsl_to_fsl_cast_inner() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 3),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_fsl_f64(
                &[Some(vec![1.0, 2.0, 3.0]), Some(vec![4.0, 5.0, 6.0])],
                3,
            )],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 3, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert_eq!(fsl.value_type(), DataType::Float32);
        let row0 = fsl.value(0);
        let vals = row0.as_primitive::<Float32Type>();
        assert_eq!(vals.value(0), 1.0);
    }

    #[test]
    fn test_wrong_dim_becomes_null() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_list_f32(&[
                Some(vec![1.0, 2.0, 3.0]),
                Some(vec![4.0, 5.0]), // wrong dim
                Some(vec![7.0, 8.0, 9.0]),
            ])],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 3, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert!(!fsl.is_null(0));
        assert!(fsl.is_null(1)); // wrong dim → null
        assert!(!fsl.is_null(2));
    }

    #[test]
    fn test_nan_rows_become_null() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_list_f32(&[
                Some(vec![1.0, 2.0, 3.0]),
                Some(vec![f32::NAN, 5.0, 6.0]),
                Some(vec![7.0, 8.0, 9.0]),
            ])],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 3, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert!(!fsl.is_null(0));
        assert!(fsl.is_null(1)); // NaN → null
        assert!(!fsl.is_null(2));
    }

    #[test]
    fn test_null_rows_preserved() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![make_list_f32(&[
                Some(vec![1.0, 2.0, 3.0]),
                None,
                Some(vec![7.0, 8.0, 9.0]),
            ])],
        )
        .unwrap();

        let expr = CastToFixedSizeListExpr::new(source_col(), 3, DataType::Float32);
        let result = eval_expr(&expr, &batch);
        let fsl = result.as_fixed_size_list();
        assert!(!fsl.is_null(0));
        assert!(fsl.is_null(1)); // was null, stays null
        assert!(!fsl.is_null(2));
    }

    #[tokio::test]
    async fn test_schema_cast_projection_noop() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1, 2])),
                Arc::new(Float32Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();

        let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem_table)).unwrap();
        let df = ctx.sql("SELECT * FROM t").await.unwrap();
        let input = df.create_physical_plan().await.unwrap();

        let result = create_schema_cast_projection(input.clone(), &schema).unwrap();
        // Should return input unchanged (same Arc)
        assert!(Arc::ptr_eq(&result, &input));
    }

    #[tokio::test]
    async fn test_schema_cast_projection_vector_cast() {
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vec",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
        ]));
        let target_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
                true,
            ),
        ]));

        let batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1, 2])),
                make_list_f32(&[Some(vec![1.0, 2.0, 3.0]), Some(vec![4.0, 5.0, 6.0])]),
            ],
        )
        .unwrap();

        let mem_table = MemTable::try_new(source_schema.clone(), vec![vec![batch]]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem_table)).unwrap();
        let df = ctx.sql("SELECT * FROM t").await.unwrap();
        let input = df.create_physical_plan().await.unwrap();

        let result_plan = create_schema_cast_projection(input, &target_schema).unwrap();
        let batches = collect(result_plan, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.len(), 1);
        let result_batch = &batches[0];

        // Verify the vec column is now FSL
        let fsl = result_batch.column(1).as_fixed_size_list();
        assert_eq!(fsl.value_length(), 3);
        let row0 = fsl.value(0);
        let vals = row0.as_primitive::<Float32Type>();
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(1), 2.0);
        assert_eq!(vals.value(2), 3.0);
    }
}

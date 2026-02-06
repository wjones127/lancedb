// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! DataFusion ExecutionPlan that wraps a [`Scannable`] data source.

use std::any::Any;
use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result as DataFusionResult, Statistics};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::arrow::SendableRecordBatchStreamExt;
use crate::data::scannable::Scannable;

/// An [`ExecutionPlan`] that reads from a [`Scannable`] data source.
///
/// This is a leaf node (no children). Rescannable sources (RecordBatch, Vec)
/// produce data on every `execute()` call. Non-rescannable sources (readers,
/// streams) return an empty stream after the first call.
#[derive(Debug)]
pub struct ScannableExec {
    source: Arc<Mutex<Box<dyn Scannable>>>,
    schema: SchemaRef,
    num_rows: Option<usize>,
    properties: PlanProperties,
}

impl ScannableExec {
    pub fn new(source: Box<dyn Scannable>) -> Self {
        let schema = source.schema();
        let num_rows = source.num_rows();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            source: Arc::new(Mutex::new(source)),
            schema,
            num_rows,
            properties,
        }
    }
}

impl DisplayAs for ScannableExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ScannableExec: num_rows={:?}", self.num_rows)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "ScannableExec")
            }
        }
    }
}

impl ExecutionPlan for ScannableExec {
    fn name(&self) -> &str {
        "ScannableExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "ScannableExec is a leaf node and cannot have children".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            source: self.source.clone(),
            schema: self.schema.clone(),
            num_rows: self.num_rows,
            properties: self.properties.clone(),
        }))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let mut source = self.source.lock().map_err(|e| {
            DataFusionError::Internal(format!("ScannableExec mutex poisoned: {}", e))
        })?;
        let stream = source.scan_as_stream();
        Ok(stream.into_df_stream())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Statistics> {
        use datafusion_common::stats::Precision;
        let num_rows = match self.num_rows {
            Some(n) => Precision::Exact(n),
            None => Precision::Absent,
        };
        Ok(Statistics {
            num_rows,
            ..Statistics::new_unknown(self.schema.as_ref())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::record_batch;
    use datafusion_common::stats::Precision;
    use datafusion_physical_plan::collect;

    use crate::arrow::{SendableRecordBatchStream as LdbStream, SimpleRecordBatchStream};

    #[tokio::test]
    async fn test_execute_record_batch() {
        let batch = record_batch!(("id", Int64, [1, 2, 3])).unwrap();
        let exec = ScannableExec::new(Box::new(batch.clone()));

        let ctx = datafusion::prelude::SessionContext::new();
        let results = collect(Arc::new(exec), ctx.task_ctx()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], batch);
    }

    #[tokio::test]
    async fn test_execute_vec_batches() {
        let batches = vec![
            record_batch!(("id", Int64, [1, 2])).unwrap(),
            record_batch!(("id", Int64, [3, 4, 5])).unwrap(),
        ];
        let exec = ScannableExec::new(Box::new(batches.clone()));

        let ctx = datafusion::prelude::SessionContext::new();
        let results = collect(Arc::new(exec), ctx.task_ctx()).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], batches[0]);
        assert_eq!(results[1], batches[1]);
    }

    #[tokio::test]
    async fn test_rescannable_multiple_executes() {
        let batch = record_batch!(("id", Int64, [1, 2, 3])).unwrap();
        let exec = Arc::new(ScannableExec::new(Box::new(batch.clone())));
        let ctx = datafusion::prelude::SessionContext::new();

        let results1 = collect(exec.clone(), ctx.task_ctx()).await.unwrap();
        assert_eq!(results1.len(), 1);
        assert_eq!(results1[0], batch);

        let results2 = collect(exec, ctx.task_ctx()).await.unwrap();
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0], batch);
    }

    #[tokio::test]
    async fn test_non_rescannable_empty_after_first() {
        let batch = record_batch!(("id", Int64, [1, 2, 3])).unwrap();
        let schema = batch.schema();
        let inner_stream = futures::stream::iter(vec![Ok(batch.clone())]);
        let stream: LdbStream = Box::pin(SimpleRecordBatchStream {
            schema: schema.clone(),
            stream: inner_stream,
        });

        let exec = Arc::new(ScannableExec::new(Box::new(stream)));
        let ctx = datafusion::prelude::SessionContext::new();

        let results1 = collect(exec.clone(), ctx.task_ctx()).await.unwrap();
        assert_eq!(results1.len(), 1);
        assert_eq!(results1[0], batch);

        let results2 = collect(exec, ctx.task_ctx()).await.unwrap();
        assert_eq!(results2.len(), 0);
    }

    #[test]
    fn test_properties() {
        let batch = record_batch!(("id", Int64, [1, 2, 3])).unwrap();
        let exec = ScannableExec::new(Box::new(batch));
        let props = exec.properties();

        assert_eq!(props.output_partitioning().partition_count(), 1);
        assert_eq!(props.emission_type, EmissionType::Incremental);
        assert_eq!(props.boundedness, Boundedness::Bounded);
    }

    #[test]
    fn test_statistics_with_num_rows() {
        let batch = record_batch!(("id", Int64, [1, 2, 3])).unwrap();
        let exec = ScannableExec::new(Box::new(batch));
        let stats = exec.partition_statistics(None).unwrap();

        assert_eq!(stats.num_rows, Precision::Exact(3));
    }

    #[test]
    fn test_statistics_without_num_rows() {
        let batch = record_batch!(("id", Int64, [1, 2, 3])).unwrap();
        let schema = batch.schema();
        let inner_stream = futures::stream::iter(vec![Ok(batch)]);
        let stream: LdbStream = Box::pin(SimpleRecordBatchStream {
            schema: schema.clone(),
            stream: inner_stream,
        });

        let exec = ScannableExec::new(Box::new(stream));
        let stats = exec.partition_statistics(None).unwrap();

        assert_eq!(stats.num_rows, Precision::Absent);
    }
}

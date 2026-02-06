// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! DataFusion ExecutionPlan for inserting data into LanceDB tables.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion_common::{DataFusionError, Result as DataFusionResult};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::StreamExt;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, InsertBuilder, WriteParams};
use lance::Dataset;
use lance_table::format::Fragment;

use crate::table::add_data::WriteProgressState;
use crate::table::dataset::DatasetConsistencyWrapper;

pub(crate) static COUNT_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(ArrowSchema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
});

fn operation_fragments(operation: &Operation) -> &[Fragment] {
    match operation {
        Operation::Append { fragments } => fragments,
        Operation::Overwrite { fragments, .. } => fragments,
        _ => &[],
    }
}

fn count_rows_from_operation(operation: &Operation) -> u64 {
    operation_fragments(operation)
        .iter()
        .map(|f| f.num_rows().unwrap_or(0) as u64)
        .sum()
}

fn operation_fragments_mut(operation: &mut Operation) -> &mut Vec<Fragment> {
    match operation {
        Operation::Append { fragments } => fragments,
        Operation::Overwrite { fragments, .. } => fragments,
        _ => panic!("Unsupported operation type for getting mutable fragments"),
    }
}

fn merge_transactions(mut transactions: Vec<Transaction>) -> Option<Transaction> {
    let mut first = transactions.pop()?;

    for txn in transactions {
        let first_fragments = operation_fragments_mut(&mut first.operation);
        let txn_fragments = operation_fragments(&txn.operation);
        first_fragments.extend_from_slice(txn_fragments);
    }

    Some(first)
}

/// Mutable state shared across partitions within a single execution.
///
/// Bundled into a single `Mutex` so that `did_reset` can be cleared once all
/// partitions (both successful and failed) have reported, allowing subsequent
/// executions of the same plan to start with fresh state.
#[derive(Debug)]
struct InsertExecState {
    partial_transactions: Vec<Transaction>,
    any_partition_failed: bool,
    /// Number of partitions that have finished (success or failure).
    completed_count: usize,
}

impl InsertExecState {
    fn new(num_partitions: usize) -> Self {
        Self {
            partial_transactions: Vec::with_capacity(num_partitions),
            any_partition_failed: false,
            completed_count: 0,
        }
    }

    fn reset(&mut self, num_partitions: usize) {
        self.partial_transactions.clear();
        self.partial_transactions.reserve(num_partitions);
        self.any_partition_failed = false;
        self.completed_count = 0;
    }
}

/// ExecutionPlan for inserting data into a native LanceDB table.
///
/// This plan executes inserts by:
/// 1. Each partition writes data independently using InsertBuilder::execute_uncommitted_stream
/// 2. The last partition to complete commits all transactions atomically
/// 3. Returns the count of inserted rows per partition
#[derive(Debug)]
pub struct InsertExec {
    ds_wrapper: DatasetConsistencyWrapper,
    dataset: Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
    write_params: WriteParams,
    properties: PlanProperties,
    exec_state: Arc<Mutex<InsertExecState>>,
    /// Ensures shared state is reset exactly once per execution, regardless of
    /// which partition calls `execute` first. Reset back to `false` after all
    /// partitions complete so subsequent executions start fresh.
    did_reset: Arc<AtomicBool>,
    progress: Option<Arc<WriteProgressState>>,
}

impl InsertExec {
    pub fn new(
        ds_wrapper: DatasetConsistencyWrapper,
        dataset: Arc<Dataset>,
        input: Arc<dyn ExecutionPlan>,
        write_params: WriteParams,
        progress: Option<Arc<WriteProgressState>>,
    ) -> Self {
        let schema = COUNT_SCHEMA.clone();
        let num_partitions = input.output_partitioning().partition_count();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(num_partitions),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            ds_wrapper,
            dataset,
            input,
            write_params,
            properties,
            exec_state: Arc::new(Mutex::new(InsertExecState::new(num_partitions))),
            did_reset: Arc::new(AtomicBool::new(false)),
            progress,
        }
    }
}

impl DisplayAs for InsertExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "InsertExec: mode={:?}", self.write_params.mode)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "InsertExec")
            }
        }
    }
}

impl ExecutionPlan for InsertExec {
    fn name(&self) -> &str {
        Self::static_name()
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
        vec![false]
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
                "InsertExec requires exactly one child".to_string(),
            ));
        }
        let mut new = Self::new(
            self.ds_wrapper.clone(),
            self.dataset.clone(),
            children[0].clone(),
            self.write_params.clone(),
            self.progress.clone(),
        );
        new.exec_state = self.exec_state.clone();
        new.did_reset = self.did_reset.clone();
        Ok(Arc::new(new))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let total_partitions = self.input.output_partitioning().partition_count();
        if self
            .did_reset
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.exec_state.lock().unwrap().reset(total_partitions);
        }
        let input_stream = self.input.execute(partition, context)?;
        let dataset = self.dataset.clone();
        let write_params = self.write_params.clone();
        let exec_state = self.exec_state.clone();
        let ds_wrapper = self.ds_wrapper.clone();
        let did_reset = self.did_reset.clone();
        let progress = self.progress.clone();

        let stream = futures::stream::once(async move {
            let input_stream = if let Some(ref progress) = progress {
                let schema = input_stream.schema();
                let progress = progress.clone();
                let mapped = input_stream.map(move |result| {
                    if let Ok(ref batch) = result {
                        // Reports in-memory Arrow array size, not on-disk bytes.
                        // TODO: plumb actual bytes written to storage from Lance.
                        progress.report(batch.num_rows(), batch.get_array_memory_size());
                    }
                    result
                });
                Box::pin(RecordBatchStreamAdapter::new(schema, mapped)) as SendableRecordBatchStream
            } else {
                input_stream
            };

            let result = InsertBuilder::new(dataset.clone())
                .with_params(&write_params)
                .execute_uncommitted_stream(input_stream)
                .await;

            let transaction = match result {
                Ok(txn) => txn,
                Err(e) => {
                    let mut state = exec_state.lock().unwrap();
                    state.any_partition_failed = true;
                    state.completed_count += 1;
                    if state.completed_count == total_partitions {
                        did_reset.store(false, Ordering::SeqCst);
                    }
                    return Err(DataFusionError::External(Box::new(e)));
                }
            };

            let num_rows = count_rows_from_operation(&transaction.operation);

            let to_commit = {
                let mut state = exec_state.lock().unwrap();
                state.partial_transactions.push(transaction);
                state.completed_count += 1;
                if state.completed_count == total_partitions {
                    did_reset.store(false, Ordering::SeqCst);
                    Some((
                        std::mem::take(&mut state.partial_transactions),
                        state.any_partition_failed,
                    ))
                } else {
                    None
                }
            };

            if let Some((transactions, any_failed)) = to_commit {
                if any_failed {
                    return Err(DataFusionError::Execution(
                        "Not committing because another partition failed".to_string(),
                    ));
                }
                if let Some(merged_txn) = merge_transactions(transactions) {
                    let new_dataset = CommitBuilder::new(dataset.clone())
                        .execute(merged_txn)
                        .await?;
                    ds_wrapper.set_latest(new_dataset).await;
                }
            }

            Ok(RecordBatch::try_new(
                COUNT_SCHEMA.clone(),
                vec![Arc::new(UInt64Array::from(vec![num_rows]))],
            )?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            COUNT_SCHEMA.clone(),
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use arrow_array::{record_batch, RecordBatchIterator};
    use datafusion::prelude::SessionContext;
    use datafusion_catalog::MemTable;
    use tempfile::tempdir;

    use crate::connect;

    /// An ExecutionPlan that wraps another plan but injects an error into one partition's stream.
    #[derive(Debug)]
    struct ErrorInjectingExec {
        input: Arc<dyn ExecutionPlan>,
        /// Which partition should produce an error.
        error_partition: usize,
        properties: PlanProperties,
    }

    impl ErrorInjectingExec {
        fn new(input: Arc<dyn ExecutionPlan>, error_partition: usize) -> Self {
            let properties = input.properties().clone();
            Self {
                input,
                error_partition,
                properties,
            }
        }
    }

    impl DisplayAs for ErrorInjectingExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "ErrorInjectingExec")
        }
    }

    impl ExecutionPlan for ErrorInjectingExec {
        fn name(&self) -> &str {
            "ErrorInjectingExec"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn schema(&self) -> SchemaRef {
            self.input.schema()
        }

        fn properties(&self) -> &PlanProperties {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self::new(
                children[0].clone(),
                self.error_partition,
            )))
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            if partition == self.error_partition {
                let schema = self.schema();
                let stream = futures::stream::once(async {
                    Err(DataFusionError::Execution(
                        "injected partition error".to_string(),
                    ))
                });
                Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
            } else {
                self.input.execute(partition, context)
            }
        }
    }

    #[tokio::test]
    async fn test_insert_via_sql() {
        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        // Create initial table
        let batch = record_batch!(("id", Int32, [1, 2, 3])).unwrap();

        let table = db
            .create_table("test_insert", batch)
            .execute()
            .await
            .unwrap();

        // Verify initial count
        assert_eq!(table.count_rows(None).await.unwrap(), 3);

        let ctx = SessionContext::new();
        let provider =
            crate::table::datafusion::BaseTableAdapter::try_new(table.base_table().clone())
                .await
                .unwrap();
        ctx.register_table("test_insert", Arc::new(provider))
            .unwrap();

        ctx.sql("INSERT INTO test_insert VALUES (4), (5), (6)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // Verify final count
        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 6);
    }

    #[tokio::test]
    async fn test_insert_overwrite_via_sql() {
        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        // Create initial table with 3 rows
        let batch = record_batch!(("id", Int32, [1, 2, 3])).unwrap();

        let table = db
            .create_table("test_overwrite", batch)
            .execute()
            .await
            .unwrap();

        assert_eq!(table.count_rows(None).await.unwrap(), 3);

        let ctx = SessionContext::new();
        let provider =
            crate::table::datafusion::BaseTableAdapter::try_new(table.base_table().clone())
                .await
                .unwrap();
        ctx.register_table("test_overwrite", Arc::new(provider))
            .unwrap();

        ctx.sql("INSERT OVERWRITE INTO test_overwrite VALUES (10), (20)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // Verify: should have 2 rows (overwritten, not appended)
        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_insert_empty_batch() {
        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        // Create initial table
        let batch = record_batch!(("id", Int32, [1, 2, 3])).unwrap();
        let table = db
            .create_table("test_empty", batch)
            .execute()
            .await
            .unwrap();

        assert_eq!(table.count_rows(None).await.unwrap(), 3);

        let ctx = SessionContext::new();
        let provider =
            crate::table::datafusion::BaseTableAdapter::try_new(table.base_table().clone())
                .await
                .unwrap();
        ctx.register_table("test_empty", Arc::new(provider))
            .unwrap();

        let source_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        // Empty batches
        let source_reader = Box::new(RecordBatchIterator::new(
            std::iter::empty::<Result<RecordBatch, arrow_schema::ArrowError>>(),
            source_schema,
        )) as Box<dyn arrow_array::RecordBatchReader + Send>;
        let source_table = db
            .create_table("empty_source", source_reader)
            .execute()
            .await
            .unwrap();
        let source_provider =
            crate::table::datafusion::BaseTableAdapter::try_new(source_table.base_table().clone())
                .await
                .unwrap();
        ctx.register_table("empty_source", Arc::new(source_provider))
            .unwrap();

        // Execute INSERT with empty source
        ctx.sql("INSERT INTO test_empty SELECT * FROM empty_source")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // Verify: should still have 3 rows (nothing inserted)
        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_insert_multiple_batches() {
        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        // Create initial table
        let batch = record_batch!(("id", Int32, [1])).unwrap();
        let schema = batch.schema();

        let table = db
            .create_table("test_multi_batch", batch)
            .execute()
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let provider =
            crate::table::datafusion::BaseTableAdapter::try_new(table.base_table().clone())
                .await
                .unwrap();
        ctx.register_table("test_multi_batch", Arc::new(provider))
            .unwrap();

        // Memtable with multiple batches and multiple partitions
        let source_table = MemTable::try_new(
            schema.clone(),
            vec![
                // Partition 0
                vec![
                    record_batch!(("id", Int32, [2, 3])).unwrap(),
                    record_batch!(("id", Int32, [4, 5])).unwrap(),
                ],
                // Partition 1
                vec![record_batch!(("id", Int32, [6, 7, 8])).unwrap()],
            ],
        )
        .unwrap();
        ctx.register_table("multi_batch_source", Arc::new(source_table))
            .unwrap();

        ctx.sql("INSERT INTO test_multi_batch SELECT * FROM multi_batch_source")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // Verify: should have 1 + 2 + 2 + 3 = 8 rows
        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 8);
    }

    #[tokio::test]
    async fn test_insert_no_commit_on_partition_failure() {
        use datafusion_catalog::TableProvider;

        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        let batch = record_batch!(("id", Int32, [1])).unwrap();
        let schema = batch.schema();

        let table = db
            .create_table("test_partition_fail", batch)
            .execute()
            .await
            .unwrap();

        assert_eq!(table.count_rows(None).await.unwrap(), 1);

        let ctx = SessionContext::new();

        // Create a source with 2 partitions, where partition 1 will error.
        let good_data = MemTable::try_new(
            schema.clone(),
            vec![
                vec![record_batch!(("id", Int32, [2, 3])).unwrap()],
                vec![record_batch!(("id", Int32, [4, 5])).unwrap()],
            ],
        )
        .unwrap();
        let good_plan = good_data.scan(&ctx.state(), None, &[], None).await.unwrap();

        // Wrap in ErrorInjectingExec to make partition 1 fail.
        let error_plan: Arc<dyn ExecutionPlan> = Arc::new(ErrorInjectingExec::new(good_plan, 1));

        // Build InsertExec directly.
        let ds_wrapper = table.dataset().unwrap().clone();
        let ds = ds_wrapper.get().await.unwrap();
        let dataset = Arc::new((*ds).clone());
        drop(ds);

        let insert_exec = InsertExec::new(
            ds_wrapper.clone(),
            dataset,
            error_plan,
            WriteParams::default(),
            None,
        );
        let insert_plan: Arc<dyn ExecutionPlan> = Arc::new(insert_exec);

        let task_ctx = ctx.task_ctx();
        let results = datafusion_physical_plan::collect(insert_plan, task_ctx).await;

        // The insert should fail because one partition errored.
        assert!(
            results.is_err(),
            "Expected insert to fail due to partition error"
        );

        // The table should still have only 1 row (no commit occurred).
        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 1);
    }

    /// Verifies that `did_reset` is properly cleared after a failed execution
    /// with partition errors, allowing subsequent executions to start fresh.
    #[tokio::test]
    async fn test_insert_state_resets_after_failure() {
        use datafusion_catalog::TableProvider;

        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        let batch = record_batch!(("id", Int32, [1])).unwrap();
        let schema = batch.schema();

        let table = db
            .create_table("test_state_reset", batch)
            .execute()
            .await
            .unwrap();

        assert_eq!(table.count_rows(None).await.unwrap(), 1);

        let ctx = SessionContext::new();

        // Execute with 2 partitions where partition 1 fails.
        let source = MemTable::try_new(
            schema.clone(),
            vec![
                vec![record_batch!(("id", Int32, [2, 3])).unwrap()],
                vec![record_batch!(("id", Int32, [4, 5])).unwrap()],
            ],
        )
        .unwrap();
        let source_plan = source.scan(&ctx.state(), None, &[], None).await.unwrap();
        let error_input: Arc<dyn ExecutionPlan> = Arc::new(ErrorInjectingExec::new(source_plan, 1));

        let ds_wrapper = table.dataset().unwrap().clone();
        let ds = ds_wrapper.get().await.unwrap();
        let dataset = Arc::new((*ds).clone());
        drop(ds);

        let insert_exec = InsertExec::new(
            ds_wrapper.clone(),
            dataset,
            error_input,
            WriteParams::default(),
            None,
        );
        let insert_plan: Arc<dyn ExecutionPlan> = Arc::new(insert_exec);

        let task_ctx = ctx.task_ctx();
        let result = datafusion_physical_plan::collect(insert_plan.clone(), task_ctx).await;
        assert!(result.is_err(), "Expected execution to fail");

        // No data should have been committed.
        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 1);

        // Verify that `did_reset` was cleared, allowing the next execution
        // to properly reset state. Without this fix, `did_reset` would remain
        // `true` and stale `any_partition_failed` state would leak.
        let insert_ref = insert_plan.as_any().downcast_ref::<InsertExec>().unwrap();
        assert!(
            !insert_ref.did_reset.load(Ordering::SeqCst),
            "did_reset should be false after all partitions completed, \
             allowing next execution to reset state"
        );

        // Verify that the failure was recorded.
        let state = insert_ref.exec_state.lock().unwrap();
        assert!(
            state.any_partition_failed,
            "any_partition_failed should be true after a partition error"
        );
        assert_eq!(
            state.completed_count, 2,
            "all partitions should have reported completion"
        );
    }

    /// Verifies that two successive successful insert executions via SQL both
    /// commit correctly, proving state doesn't leak between runs.
    #[tokio::test]
    async fn test_insert_two_successive_executions() {
        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        let batch = record_batch!(("id", Int32, [1])).unwrap();

        let table = db
            .create_table("test_successive", batch)
            .execute()
            .await
            .unwrap();

        assert_eq!(table.count_rows(None).await.unwrap(), 1);

        let ctx = SessionContext::new();
        let provider =
            crate::table::datafusion::BaseTableAdapter::try_new(table.base_table().clone())
                .await
                .unwrap();
        ctx.register_table("test_successive", Arc::new(provider))
            .unwrap();

        // First insert
        ctx.sql("INSERT INTO test_successive VALUES (2), (3)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 3);

        // Second insert on the same registered table
        ctx.sql("INSERT INTO test_successive VALUES (4), (5)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_insert_with_progress() {
        use crate::table::add_data::{WriteProgress, WriteProgressState};
        use datafusion_catalog::TableProvider;

        let tmp_dir = tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let db = connect(uri).execute().await.unwrap();

        let batch = record_batch!(("id", Int32, [1])).unwrap();
        let schema = batch.schema();

        let table = db
            .create_table("test_progress", batch)
            .execute()
            .await
            .unwrap();

        let ctx = SessionContext::new();

        // Multi-batch, multi-partition source
        let source = MemTable::try_new(
            schema.clone(),
            vec![
                vec![
                    record_batch!(("id", Int32, [2, 3])).unwrap(),
                    record_batch!(("id", Int32, [4, 5])).unwrap(),
                ],
                vec![record_batch!(("id", Int32, [6, 7, 8])).unwrap()],
            ],
        )
        .unwrap();
        let source_plan = source.scan(&ctx.state(), None, &[], None).await.unwrap();

        let updates: Arc<Mutex<Vec<WriteProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_clone = updates.clone();
        let progress = Arc::new(WriteProgressState::new(Arc::new(move |p| {
            updates_clone.lock().unwrap().push(p);
        })));

        let ds_wrapper = table.dataset().unwrap().clone();
        let ds = ds_wrapper.get().await.unwrap();
        let dataset = Arc::new((*ds).clone());
        drop(ds);

        let write_params = WriteParams {
            mode: lance::dataset::WriteMode::Append,
            ..Default::default()
        };

        let insert_exec = InsertExec::new(
            ds_wrapper.clone(),
            dataset,
            source_plan,
            write_params,
            Some(progress),
        );
        let insert_plan: Arc<dyn ExecutionPlan> = Arc::new(insert_exec);

        let task_ctx = ctx.task_ctx();
        datafusion_physical_plan::collect(insert_plan, task_ctx)
            .await
            .unwrap();

        table.checkout_latest().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 8);

        let updates = updates.lock().unwrap();
        assert!(!updates.is_empty(), "should have received progress updates");
        let last = updates.last().unwrap();
        assert!(last.rows_written > 0);
        assert!(last.bytes_written > 0);
        assert!(last.elapsed.as_nanos() > 0);
    }
}

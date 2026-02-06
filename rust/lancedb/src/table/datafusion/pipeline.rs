// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Pipeline assembly for the insert execution plan.
//!
//! Builds the processing pipeline:
//! `ScannableExec → EmbeddingProjection → SchemaCastProjection → InsertExec`

use std::sync::Arc;

use datafusion_physical_plan::ExecutionPlan;

use crate::embeddings::{EmbeddingDefinition, EmbeddingFunction, EmbeddingRegistry};
use crate::table::{AddDataMode, BaseTable, ColumnKind, TableDefinition};
use crate::Error;

use super::cast_vector::create_schema_cast_projection;
use super::embedding_udf::build_embedding_projection;

/// Resolves embedding functions from a [`TableDefinition`] and [`EmbeddingRegistry`].
///
/// Returns a list of (definition, function) pairs for each embedding column defined
/// in the table. Returns an empty vec if no embedding columns are defined.
pub fn resolve_embeddings(
    table_def: &TableDefinition,
    registry: &dyn EmbeddingRegistry,
) -> crate::Result<Vec<(EmbeddingDefinition, Arc<dyn EmbeddingFunction>)>> {
    let mut embeddings = Vec::new();
    for cd in &table_def.column_definitions {
        if let ColumnKind::Embedding(embedding_def) = &cd.kind {
            match registry.get(&embedding_def.embedding_name) {
                Some(func) => {
                    embeddings.push((embedding_def.clone(), func));
                }
                None => {
                    return Err(Error::EmbeddingFunctionNotFound {
                        name: embedding_def.embedding_name.clone(),
                        reason: format!(
                            "Table was defined with an embedding column `{}` but no embedding \
                             function was found with that name within the registry.",
                            embedding_def.embedding_name
                        ),
                    });
                }
            }
        }
    }
    Ok(embeddings)
}

/// Builds the processing pipeline that sits between a data source and an insert node.
///
/// Applies, in order:
/// 1. Embedding projection (if the table has embedding columns and a registry is provided)
/// 2. Schema cast projection (adapts input schema to the table's target schema)
///
/// In [`AddDataMode::Overwrite`] mode, schema casting is skipped because the new data
/// defines the table's schema.
///
/// If the table definition is unavailable (e.g. [`RemoteTable`]), embedding resolution
/// is skipped but schema casting is still applied.
pub async fn build_processing_pipeline(
    input: Arc<dyn ExecutionPlan>,
    table: &dyn BaseTable,
    embedding_registry: Option<&Arc<dyn EmbeddingRegistry>>,
    mode: &AddDataMode,
) -> crate::Result<Arc<dyn ExecutionPlan>> {
    let mut plan = input;

    // Try to resolve embeddings from the table definition.
    // RemoteTable::table_definition() returns NotSupported, which we skip since
    // remote tables handle embeddings server-side. All other errors are propagated.
    if let Some(registry) = embedding_registry {
        match table.table_definition().await {
            Ok(table_def) => {
                let embeddings = resolve_embeddings(&table_def, registry.as_ref())?;
                if !embeddings.is_empty() {
                    plan = build_embedding_projection(plan, &embeddings).map_err(|e| {
                        Error::Runtime {
                            message: e.to_string(),
                        }
                    })?;
                }
            }
            Err(Error::NotSupported { .. }) => {
                // Remote tables don't support table_definition(); skip embeddings.
            }
            Err(e) => return Err(e),
        }
    }

    // In overwrite mode the new data defines the schema, so skip casting.
    if !matches!(mode, AddDataMode::Overwrite) {
        let target_schema = table.schema().await?;
        plan = create_schema_cast_projection(plan, &target_schema).map_err(|e| Error::Runtime {
            message: e.to_string(),
        })?;
    }

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MemoryRegistry;
    use crate::table::{ColumnDefinition, ColumnKind};
    use crate::test_utils::embeddings::MockEmbed;
    use arrow_schema::{DataType, Field, Schema};

    use arrow_array::record_batch;
    use lance_datafusion::exec::OneShotExec;

    /// BaseTable wrapper that delegates everything to an inner table
    /// except `table_definition()`, which returns a configurable error.
    #[derive(Debug)]
    struct FailingTableDef {
        inner: Arc<dyn crate::table::BaseTable>,
        error_message: String,
    }

    impl std::fmt::Display for FailingTableDef {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FailingTableDef({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl crate::table::BaseTable for FailingTableDef {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn name(&self) -> &str {
            self.inner.name()
        }
        fn namespace(&self) -> &[String] {
            self.inner.namespace()
        }
        fn id(&self) -> &str {
            self.inner.id()
        }
        async fn schema(&self) -> crate::Result<arrow_schema::SchemaRef> {
            self.inner.schema().await
        }
        async fn table_definition(&self) -> crate::Result<crate::table::TableDefinition> {
            Err(Error::Runtime {
                message: self.error_message.clone(),
            })
        }
        async fn count_rows(&self, f: Option<crate::table::Filter>) -> crate::Result<usize> {
            self.inner.count_rows(f).await
        }
        async fn create_plan(
            &self,
            q: &crate::table::AnyQuery,
            o: crate::query::QueryExecutionOptions,
        ) -> crate::Result<Arc<dyn ExecutionPlan>> {
            self.inner.create_plan(q, o).await
        }
        async fn query(
            &self,
            q: &crate::table::AnyQuery,
            o: crate::query::QueryExecutionOptions,
        ) -> crate::Result<lance::dataset::scanner::DatasetRecordBatchStream> {
            self.inner.query(q, o).await
        }
        async fn analyze_plan(
            &self,
            q: &crate::table::AnyQuery,
            o: crate::query::QueryExecutionOptions,
        ) -> crate::Result<String> {
            self.inner.analyze_plan(q, o).await
        }
        async fn add(
            &self,
            a: crate::table::AddDataBuilder,
        ) -> crate::Result<crate::table::AddResult> {
            self.inner.add(a).await
        }
        async fn delete(&self, p: &str) -> crate::Result<crate::table::DeleteResult> {
            self.inner.delete(p).await
        }
        async fn update(
            &self,
            u: crate::table::UpdateBuilder,
        ) -> crate::Result<crate::table::UpdateResult> {
            self.inner.update(u).await
        }
        async fn create_index(&self, i: crate::index::IndexBuilder) -> crate::Result<()> {
            self.inner.create_index(i).await
        }
        async fn list_indices(&self) -> crate::Result<Vec<crate::index::IndexConfig>> {
            self.inner.list_indices().await
        }
        async fn drop_index(&self, n: &str) -> crate::Result<()> {
            self.inner.drop_index(n).await
        }
        async fn prewarm_index(&self, n: &str) -> crate::Result<()> {
            self.inner.prewarm_index(n).await
        }
        async fn index_stats(
            &self,
            n: &str,
        ) -> crate::Result<Option<crate::index::IndexStatistics>> {
            self.inner.index_stats(n).await
        }
        async fn merge_insert(
            &self,
            p: crate::table::merge::MergeInsertBuilder,
            d: Box<dyn arrow_array::RecordBatchReader + Send>,
        ) -> crate::Result<crate::table::MergeResult> {
            self.inner.merge_insert(p, d).await
        }
        async fn tags(&self) -> crate::Result<Box<dyn crate::table::Tags + '_>> {
            self.inner.tags().await
        }
        async fn optimize(
            &self,
            a: crate::table::OptimizeAction,
        ) -> crate::Result<crate::table::OptimizeStats> {
            self.inner.optimize(a).await
        }
        async fn add_columns(
            &self,
            t: lance::dataset::NewColumnTransform,
            c: Option<Vec<String>>,
        ) -> crate::Result<crate::table::AddColumnsResult> {
            self.inner.add_columns(t, c).await
        }
        async fn alter_columns(
            &self,
            a: &[lance::dataset::ColumnAlteration],
        ) -> crate::Result<crate::table::AlterColumnsResult> {
            self.inner.alter_columns(a).await
        }
        async fn drop_columns(&self, c: &[&str]) -> crate::Result<crate::table::DropColumnsResult> {
            self.inner.drop_columns(c).await
        }
        async fn version(&self) -> crate::Result<u64> {
            self.inner.version().await
        }
        async fn checkout(&self, v: u64) -> crate::Result<()> {
            self.inner.checkout(v).await
        }
        async fn checkout_tag(&self, t: &str) -> crate::Result<()> {
            self.inner.checkout_tag(t).await
        }
        async fn checkout_latest(&self) -> crate::Result<()> {
            self.inner.checkout_latest().await
        }
        async fn restore(&self) -> crate::Result<()> {
            self.inner.restore().await
        }
        async fn list_versions(&self) -> crate::Result<Vec<lance::dataset::Version>> {
            self.inner.list_versions().await
        }
        async fn uri(&self) -> crate::Result<String> {
            self.inner.uri().await
        }
        async fn storage_options(&self) -> Option<std::collections::HashMap<String, String>> {
            self.inner.storage_options().await
        }
        async fn wait_for_index(&self, n: &[&str], t: std::time::Duration) -> crate::Result<()> {
            self.inner.wait_for_index(n, t).await
        }
        async fn stats(&self) -> crate::Result<crate::table::TableStatistics> {
            self.inner.stats().await
        }
        async fn create_insert_exec(
            &self,
            i: Arc<dyn ExecutionPlan>,
            w: lance::dataset::WriteParams,
        ) -> crate::Result<Arc<dyn ExecutionPlan>> {
            self.inner.create_insert_exec(i, w).await
        }
    }

    #[test]
    fn test_resolve_embeddings_none() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let table_def = TableDefinition::new_from_schema(schema);
        let registry = MemoryRegistry::new();

        let result = resolve_embeddings(&table_def, &registry).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_embeddings_found() {
        let embedding_def = EmbeddingDefinition::new("text", "mock", Some("text_embedding"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "text_embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
        ]));
        let table_def = TableDefinition::new(
            schema,
            vec![
                ColumnDefinition {
                    kind: ColumnKind::Physical,
                },
                ColumnDefinition {
                    kind: ColumnKind::Embedding(embedding_def),
                },
            ],
        );

        let registry = MemoryRegistry::new();
        let mock: Arc<dyn EmbeddingFunction> = Arc::new(MockEmbed::new("mock", 4));
        registry.register("mock", mock).unwrap();

        let result = resolve_embeddings(&table_def, &registry).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.source_column, "text");
    }

    #[test]
    fn test_resolve_embeddings_missing() {
        let embedding_def = EmbeddingDefinition::new("text", "nonexistent", Some("text_embedding"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "text_embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
        ]));
        let table_def = TableDefinition::new(
            schema,
            vec![
                ColumnDefinition {
                    kind: ColumnKind::Physical,
                },
                ColumnDefinition {
                    kind: ColumnKind::Embedding(embedding_def),
                },
            ],
        );

        let registry = MemoryRegistry::new();
        let result = resolve_embeddings(&table_def, &registry);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::EmbeddingFunctionNotFound { .. }
        ));
    }

    #[tokio::test]
    async fn test_pipeline_propagates_table_definition_error() {
        let conn = crate::connect("memory://").execute().await.unwrap();
        let batch = record_batch!(("id", Int64, [1, 2])).unwrap();
        let table = conn
            .create_table("test_err", batch.clone())
            .execute()
            .await
            .unwrap();

        let failing = FailingTableDef {
            inner: table.base_table().clone(),
            error_message: "simulated I/O error".into(),
        };

        let registry: Arc<dyn EmbeddingRegistry> = Arc::new(MemoryRegistry::new());
        let input: Arc<dyn ExecutionPlan> = Arc::new(OneShotExec::new(Box::pin(
            datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                batch.schema(),
                futures::stream::iter(vec![Ok(batch)]),
            ),
        )));

        let result =
            build_processing_pipeline(input, &failing, Some(&registry), &AddDataMode::Append).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("simulated I/O error"),
            "expected error to propagate, got: {}",
            err_msg,
        );
    }

    #[tokio::test]
    async fn test_pipeline_skips_not_supported() {
        // RemoteTable returns NotSupported from table_definition().
        // Verify the pipeline still succeeds (skips embeddings).
        let table = crate::Table::new_with_handler("test_ns", move |request| {
            let path = request.url().path();
            if path == "/v1/table/test_ns/describe/" {
                return http::Response::builder()
                    .status(200)
                    .body(
                        r#"{"version": 1, "schema": {"fields": [{"name": "id", "type": {"type": "int32"}, "nullable": true}]}}"#
                            .to_string(),
                    )
                    .unwrap();
            }
            panic!("Unexpected request: {}", path);
        });

        let batch = record_batch!(("id", Int32, [1, 2])).unwrap();
        let registry: Arc<dyn EmbeddingRegistry> = Arc::new(MemoryRegistry::new());
        let input: Arc<dyn ExecutionPlan> = Arc::new(OneShotExec::new(Box::pin(
            datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                batch.schema(),
                futures::stream::iter(vec![Ok(batch)]),
            ),
        )));

        let result = build_processing_pipeline(
            input,
            table.base_table().as_ref(),
            Some(&registry),
            &AddDataMode::Append,
        )
        .await;

        assert!(
            result.is_ok(),
            "NotSupported should be skipped, got: {:?}",
            result.err()
        );
    }
}

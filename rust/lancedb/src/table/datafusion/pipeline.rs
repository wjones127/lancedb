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
    // RemoteTable::table_definition() returns NotSupported, so we silently skip.
    if let Some(registry) = embedding_registry {
        if let Ok(table_def) = table.table_definition().await {
            let embeddings = resolve_embeddings(&table_def, registry.as_ref())?;
            if !embeddings.is_empty() {
                plan =
                    build_embedding_projection(plan, &embeddings).map_err(|e| Error::Runtime {
                        message: e.to_string(),
                    })?;
            }
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
}

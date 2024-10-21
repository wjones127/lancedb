[vectordb](../README.md) / [Exports](../modules.md) / EmbeddingFunction

# Interface: EmbeddingFunction\<T\>

An embedding function that automatically creates vector representation for a given column.

## Type parameters

| Name |
| :------ |
| `T` |

## Implemented by

- [`OpenAIEmbeddingFunction`](../classes/OpenAIEmbeddingFunction.md)

## Table of contents

### Properties

- [destColumn](EmbeddingFunction.md#destcolumn)
- [embed](EmbeddingFunction.md#embed)
- [embeddingDataType](EmbeddingFunction.md#embeddingdatatype)
- [embeddingDimension](EmbeddingFunction.md#embeddingdimension)
- [excludeSource](EmbeddingFunction.md#excludesource)
- [sourceColumn](EmbeddingFunction.md#sourcecolumn)

## Properties

### destColumn

• `Optional` **destColumn**: `string`

The name of the column that will contain the embedding

By default this is "vector"

#### Defined in

[embedding/embedding_function.ts:49](https://github.com/lancedb/lancedb/blob/92179835/node/src/embedding/embedding_function.ts#L49)

___

### embed

• **embed**: (`data`: `T`[]) => `Promise`\<`number`[][]\>

#### Type declaration

▸ (`data`): `Promise`\<`number`[][]\>

Creates a vector representation for the given values.

##### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `T`[] |

##### Returns

`Promise`\<`number`[][]\>

#### Defined in

[embedding/embedding_function.ts:62](https://github.com/lancedb/lancedb/blob/92179835/node/src/embedding/embedding_function.ts#L62)

___

### embeddingDataType

• `Optional` **embeddingDataType**: `Float`\<`Floats`\>

The data type of the embedding

The embedding function should return `number`.  This will be converted into
an Arrow float array.  By default this will be Float32 but this property can
be used to control the conversion.

#### Defined in

[embedding/embedding_function.ts:33](https://github.com/lancedb/lancedb/blob/92179835/node/src/embedding/embedding_function.ts#L33)

___

### embeddingDimension

• `Optional` **embeddingDimension**: `number`

The dimension of the embedding

This is optional, normally this can be determined by looking at the results of
`embed`.  If this is not specified, and there is an attempt to apply the embedding
to an empty table, then that process will fail.

#### Defined in

[embedding/embedding_function.ts:42](https://github.com/lancedb/lancedb/blob/92179835/node/src/embedding/embedding_function.ts#L42)

___

### excludeSource

• `Optional` **excludeSource**: `boolean`

Should the source column be excluded from the resulting table

By default the source column is included.  Set this to true and
only the embedding will be stored.

#### Defined in

[embedding/embedding_function.ts:57](https://github.com/lancedb/lancedb/blob/92179835/node/src/embedding/embedding_function.ts#L57)

___

### sourceColumn

• **sourceColumn**: `string`

The name of the column that will be used as input for the Embedding Function.

#### Defined in

[embedding/embedding_function.ts:24](https://github.com/lancedb/lancedb/blob/92179835/node/src/embedding/embedding_function.ts#L24)

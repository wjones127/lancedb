[vectordb](../README.md) / [Exports](../modules.md) / Query

# Class: Query\<T\>

A builder for nearest neighbor queries for LanceDB.

## Type parameters

| Name | Type |
| :------ | :------ |
| `T` | `number`[] |

## Table of contents

### Constructors

- [constructor](Query.md#constructor)

### Properties

- [\_embeddings](Query.md#_embeddings)
- [\_fastSearch](Query.md#_fastsearch)
- [\_filter](Query.md#_filter)
- [\_limit](Query.md#_limit)
- [\_metricType](Query.md#_metrictype)
- [\_nprobes](Query.md#_nprobes)
- [\_prefilter](Query.md#_prefilter)
- [\_query](Query.md#_query)
- [\_queryVector](Query.md#_queryvector)
- [\_refineFactor](Query.md#_refinefactor)
- [\_select](Query.md#_select)
- [\_tbl](Query.md#_tbl)
- [where](Query.md#where)

### Methods

- [execute](Query.md#execute)
- [fastSearch](Query.md#fastsearch)
- [filter](Query.md#filter)
- [isElectron](Query.md#iselectron)
- [limit](Query.md#limit)
- [metricType](Query.md#metrictype)
- [nprobes](Query.md#nprobes)
- [prefilter](Query.md#prefilter)
- [refineFactor](Query.md#refinefactor)
- [select](Query.md#select)

## Constructors

### constructor

• **new Query**\<`T`\>(`query?`, `tbl?`, `embeddings?`)

#### Type parameters

| Name | Type |
| :------ | :------ |
| `T` | `number`[] |

#### Parameters

| Name | Type |
| :------ | :------ |
| `query?` | `T` |
| `tbl?` | `any` |
| `embeddings?` | [`EmbeddingFunction`](../interfaces/EmbeddingFunction.md)\<`T`\> |

#### Defined in

[query.ts:39](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L39)

## Properties

### \_embeddings

• `Protected` `Optional` `Readonly` **\_embeddings**: [`EmbeddingFunction`](../interfaces/EmbeddingFunction.md)\<`T`\>

#### Defined in

[query.ts:37](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L37)

___

### \_fastSearch

• `Private` **\_fastSearch**: `boolean`

#### Defined in

[query.ts:36](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L36)

___

### \_filter

• `Private` `Optional` **\_filter**: `string`

#### Defined in

[query.ts:33](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L33)

___

### \_limit

• `Private` `Optional` **\_limit**: `number`

#### Defined in

[query.ts:29](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L29)

___

### \_metricType

• `Private` `Optional` **\_metricType**: [`MetricType`](../enums/MetricType.md)

#### Defined in

[query.ts:34](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L34)

___

### \_nprobes

• `Private` **\_nprobes**: `number`

#### Defined in

[query.ts:31](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L31)

___

### \_prefilter

• `Private` **\_prefilter**: `boolean`

#### Defined in

[query.ts:35](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L35)

___

### \_query

• `Private` `Optional` `Readonly` **\_query**: `T`

#### Defined in

[query.ts:26](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L26)

___

### \_queryVector

• `Private` `Optional` **\_queryVector**: `number`[]

#### Defined in

[query.ts:28](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L28)

___

### \_refineFactor

• `Private` `Optional` **\_refineFactor**: `number`

#### Defined in

[query.ts:30](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L30)

___

### \_select

• `Private` `Optional` **\_select**: `string`[]

#### Defined in

[query.ts:32](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L32)

___

### \_tbl

• `Private` `Optional` `Readonly` **\_tbl**: `any`

#### Defined in

[query.ts:27](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L27)

___

### where

• **where**: (`value`: `string`) => [`Query`](Query.md)\<`T`\>

#### Type declaration

▸ (`value`): [`Query`](Query.md)\<`T`\>

A filter statement to be applied to this query.

##### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | `string` | A filter in the same format used by a sql WHERE clause. |

##### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:90](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L90)

## Methods

### execute

▸ **execute**\<`T`\>(): `Promise`\<`T`[]\>

Execute the query and return the results as an Array of Objects

#### Type parameters

| Name | Type |
| :------ | :------ |
| `T` | `Record`\<`string`, `unknown`\> |

#### Returns

`Promise`\<`T`[]\>

#### Defined in

[query.ts:127](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L127)

___

### fastSearch

▸ **fastSearch**(`value`): [`Query`](Query.md)\<`T`\>

Skip searching un-indexed data. This can make search faster, but will miss
any data that is not yet indexed.

#### Parameters

| Name | Type |
| :------ | :------ |
| `value` | `boolean` |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:119](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L119)

___

### filter

▸ **filter**(`value`): [`Query`](Query.md)\<`T`\>

A filter statement to be applied to this query.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | `string` | A filter in the same format used by a sql WHERE clause. |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:85](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L85)

___

### isElectron

▸ `Private` **isElectron**(): `boolean`

#### Returns

`boolean`

#### Defined in

[query.ts:155](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L155)

___

### limit

▸ **limit**(`value`): [`Query`](Query.md)\<`T`\>

Sets the number of results that will be returned
default value is 10

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | `number` | number of results |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:58](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L58)

___

### metricType

▸ **metricType**(`value`): [`Query`](Query.md)\<`T`\>

The MetricType used for this Query.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | [`MetricType`](../enums/MetricType.md) | The metric to the. |

#### Returns

[`Query`](Query.md)\<`T`\>

**`See`**

MetricType for the different options

#### Defined in

[query.ts:105](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L105)

___

### nprobes

▸ **nprobes**(`value`): [`Query`](Query.md)\<`T`\>

The number of probes used. A higher number makes search more accurate but also slower.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | `number` | The number of probes used. |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:76](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L76)

___

### prefilter

▸ **prefilter**(`value`): [`Query`](Query.md)\<`T`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `value` | `boolean` |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:110](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L110)

___

### refineFactor

▸ **refineFactor**(`value`): [`Query`](Query.md)\<`T`\>

Refine the results by reading extra elements and re-ranking them in memory.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | `number` | refine factor to use in this query. |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:67](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L67)

___

### select

▸ **select**(`value`): [`Query`](Query.md)\<`T`\>

Return only the specified columns.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `value` | `string`[] | Only select the specified columns. If not specified, all columns will be returned. |

#### Returns

[`Query`](Query.md)\<`T`\>

#### Defined in

[query.ts:96](https://github.com/lancedb/lancedb/blob/92179835/node/src/query.ts#L96)

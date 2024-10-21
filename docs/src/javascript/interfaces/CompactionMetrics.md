[vectordb](../README.md) / [Exports](../modules.md) / CompactionMetrics

# Interface: CompactionMetrics

## Table of contents

### Properties

- [filesAdded](CompactionMetrics.md#filesadded)
- [filesRemoved](CompactionMetrics.md#filesremoved)
- [fragmentsAdded](CompactionMetrics.md#fragmentsadded)
- [fragmentsRemoved](CompactionMetrics.md#fragmentsremoved)

## Properties

### filesAdded

• **filesAdded**: `number`

The number of files added. This is typically equal to the number of
fragments added.

#### Defined in

[index.ts:1273](https://github.com/lancedb/lancedb/blob/92179835/node/src/index.ts#L1273)

___

### filesRemoved

• **filesRemoved**: `number`

The number of files that were removed. Each fragment may have more than one
file.

#### Defined in

[index.ts:1268](https://github.com/lancedb/lancedb/blob/92179835/node/src/index.ts#L1268)

___

### fragmentsAdded

• **fragmentsAdded**: `number`

The number of new fragments that were created.

#### Defined in

[index.ts:1263](https://github.com/lancedb/lancedb/blob/92179835/node/src/index.ts#L1263)

___

### fragmentsRemoved

• **fragmentsRemoved**: `number`

The number of fragments that were removed.

#### Defined in

[index.ts:1259](https://github.com/lancedb/lancedb/blob/92179835/node/src/index.ts#L1259)

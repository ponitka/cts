export const description = `writeTexture + copyBufferToTexture + copyTextureToBuffer operation tests.

* TODO: add another initMethod which renders the texture

* TODO: check that CopyT2B doesn't overwrite any other bytes

* TODO: because of expectContests 4-bytes alignment we don't test CopyT2B with buffer size not divisible by 4

* TODO: investigate float issues (rg11b10float format tests)

* TODO: investigate snorm issues (rgba8snorm format tests)
`;

import { params, poptions } from '../../../common/framework/params_builder.js';
import { makeTestGroup } from '../../../common/framework/test_group.js';
import { assert, unreachable } from '../../../common/framework/util/util.js';
import { kTextureFormatInfo, kTextureFormats } from '../../capability_info.js';
import { GPUTest } from '../../gpu_test.js';
import { align } from '../../util/math.js';
import { getTextureCopyLayout, TextureCopyLayout } from '../../util/texture/layout.js';

interface TextureCopyViewWithRequiredOrigin {
  texture: GPUTexture;
  mipLevel: number | undefined;
  origin: Required<GPUOrigin3DDict>;
}

class CopyBetweenLinearDataAndTextureTest extends GPUTest {
  bytesInACompleteRow(copyWidth: number, format: GPUTextureFormat): number {
    assert(copyWidth % kTextureFormatInfo[format].blockWidth! === 0);
    return (
      (kTextureFormatInfo[format].bytesPerBlock! * copyWidth) /
      kTextureFormatInfo[format].blockWidth!
    );
  }

  requiredBytesInCopy(
    layout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    copyExtent: GPUExtent3DDict
  ): number {
    assert(layout.rowsPerImage! % kTextureFormatInfo[format].blockHeight! === 0);
    assert(copyExtent.height % kTextureFormatInfo[format].blockHeight! === 0);
    assert(copyExtent.width % kTextureFormatInfo[format].blockWidth! === 0);
    if (copyExtent.width === 0 || copyExtent.height === 0 || copyExtent.depth === 0) {
      return 0;
    } else {
      const texelBlockRowsPerImage = layout.rowsPerImage! / kTextureFormatInfo[format].blockHeight!;
      const bytesPerImage = layout.bytesPerRow * texelBlockRowsPerImage;
      const bytesInLastSlice =
        layout.bytesPerRow * (copyExtent.height / kTextureFormatInfo[format].blockHeight! - 1) +
        (copyExtent.width / kTextureFormatInfo[format].blockWidth!) *
          kTextureFormatInfo[format].bytesPerBlock!;
      return bytesPerImage * (copyExtent.depth - 1) + bytesInLastSlice;
    }
  }

  // Offset for a particular texel in the linear texture data
  getTexelOffsetInBytes(
    textureDataLayout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    texel: Required<GPUOrigin3DDict>,
    origin: Required<GPUOrigin3DDict> = { x: 0, y: 0, z: 0 }
  ): number {
    const { offset, bytesPerRow, rowsPerImage } = textureDataLayout;
    const info = kTextureFormatInfo[format];

    assert(texel.x >= origin.x && texel.y >= origin.y && texel.z >= origin.z);
    assert(rowsPerImage! % info.blockHeight! === 0);
    assert(texel.x % info.blockWidth! === 0);
    assert(texel.y % info.blockHeight! === 0);
    assert(origin.x % info.blockWidth! === 0);
    assert(origin.y % info.blockHeight! === 0);

    const bytesPerImage = (rowsPerImage! / info.blockHeight!) * bytesPerRow;

    return (
      offset! +
      (texel.z - origin.z) * bytesPerImage +
      ((texel.y - origin.y) / info.blockHeight!) * bytesPerRow +
      ((texel.x - origin.x) / info.blockWidth!) * info.bytesPerBlock!
    );
  }

  *iterateBlockRows(
    size: GPUExtent3DDict,
    origin: Required<GPUOrigin3DDict>,
    format: GPUTextureFormat
  ): Generator<Required<GPUOrigin3DDict>> {
    const info = kTextureFormatInfo[format];
    assert(size.height % info.blockHeight! === 0);
    for (let y = 0; y < size.height / info.blockHeight!; ++y) {
      for (let z = 0; z < size.depth; ++z) {
        yield {
          x: origin.x,
          y: origin.y + y * info.blockHeight!,
          z: origin.z + z,
        };
      }
    }
  }

  generateData(byteSize: number): Uint8Array {
    const arr = new Uint8Array(byteSize);
    for (let i = 0; i < byteSize; ++i) {
      arr[i] = (i ** 3 + i) % 251;
    }
    return arr;
  }

  // Put data into texture with an appropriate method.
  initTexture(
    textureCopyView: GPUTextureCopyView,
    textureDataLayout: GPUTextureDataLayout,
    copySize: GPUExtent3D,
    partialData: Uint8Array,
    method: string
  ): void {
    switch (method) {
      case 'WriteTexture': {
        this.device.defaultQueue.writeTexture(
          textureCopyView,
          partialData,
          textureDataLayout,
          copySize
        );

        break;
      }
      case 'CopyB2T': {
        const buffer = this.device.createBuffer({
          mappedAtCreation: true,
          size: align(partialData.byteLength, 4),
          usage: GPUBufferUsage.COPY_SRC,
        });
        new Uint8Array(buffer.getMappedRange()).set(partialData);
        buffer.unmap();

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToTexture({ buffer, ...textureDataLayout }, textureCopyView, copySize);
        this.device.defaultQueue.submit([encoder.finish()]);

        break;
      }
      default:
        unreachable();
    }
  }

  checkData(
    { texture, mipLevel, origin }: TextureCopyViewWithRequiredOrigin,
    textureDataLayout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    checkSize: GPUExtent3DDict,
    expected: Uint8Array
  ): void {
    const buffer = this.device.createBuffer({
      size: expected.byteLength + 3, // the additional 3 bytes are needed for expectContents
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture, mipLevel, origin },
      { buffer, ...textureDataLayout },
      checkSize
    );
    this.device.defaultQueue.submit([encoder.finish()]);

    for (const texel of this.iterateBlockRows(checkSize, origin, format)) {
      const rowOffset = this.getTexelOffsetInBytes(textureDataLayout, format, texel, origin);
      const rowLength =
        (checkSize.width / kTextureFormatInfo[format].blockWidth!) *
        kTextureFormatInfo[format].bytesPerBlock!;
      this.expectContents(buffer, expected.slice(rowOffset, rowOffset + rowLength), rowOffset);
    }
  }

  getFullData(
    { texture, mipLevel }: { texture: GPUTexture; mipLevel: number | undefined },
    fullTextureCopyLayout: TextureCopyLayout
  ): GPUBuffer {
    const { mipSize, byteLength, bytesPerRow, rowsPerImage } = fullTextureCopyLayout;
    const buffer = this.device.createBuffer({
      size: align(byteLength, 4), // this is necessary because we need to copy and map data from this buffer
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture, mipLevel },
      { buffer, bytesPerRow, rowsPerImage },
      mipSize
    );
    this.device.defaultQueue.submit([encoder.finish()]);

    return buffer;
  }

  updateFullData(
    fullTextureCopyLayout: TextureCopyLayout,
    texturePartialDataLayout: GPUTextureDataLayout,
    copySize: GPUExtent3DDict,
    origin: Required<GPUOrigin3DDict>,
    format: GPUTextureFormat,
    fullData: Uint8Array,
    partialData: Uint8Array
  ): void {
    for (const texel of this.iterateBlockRows(copySize, origin, format)) {
      const partialDataOffset = this.getTexelOffsetInBytes(
        texturePartialDataLayout,
        format,
        texel,
        origin
      );
      const fullDataOffset = this.getTexelOffsetInBytes(
        {
          bytesPerRow: fullTextureCopyLayout.bytesPerRow,
          rowsPerImage: fullTextureCopyLayout.rowsPerImage,
          offset: 0,
        },
        format,
        texel
      );
      const rowLength =
        (copySize.width / kTextureFormatInfo[format].blockWidth!) *
        kTextureFormatInfo[format].bytesPerBlock!;
      for (let b = 0; b < rowLength; ++b) {
        fullData[fullDataOffset + b] = partialData[partialDataOffset + b];
      }
    }
  }

  fullCheck(
    { texture, mipLevel, origin }: TextureCopyViewWithRequiredOrigin,
    fullTextureCopyLayout: TextureCopyLayout,
    texturePartialDataLayout: GPUTextureDataLayout,
    copySize: GPUExtent3DDict,
    format: GPUTextureFormat,
    fullData: GPUBuffer,
    partialData: Uint8Array
  ): void {
    const { mipSize, bytesPerRow, rowsPerImage, byteLength } = fullTextureCopyLayout;
    const { dst, begin, end } = this.createAlignedCopyForMapRead(fullData, byteLength, 0);

    // We add an eventual async expectation which will update the full data and then add
    // other eventual async expectations to ensure it will be correct.
    this.eventualAsyncExpectation(async () => {
      await dst.mapAsync(GPUMapMode.READ);
      const actual = new Uint8Array(dst.getMappedRange()).slice(begin, end);
      this.updateFullData(
        fullTextureCopyLayout,
        texturePartialDataLayout,
        copySize,
        origin,
        format,
        actual,
        partialData
      );
      this.checkData(
        { texture, mipLevel, origin: { x: 0, y: 0, z: 0 } },
        { bytesPerRow, rowsPerImage, offset: 0 },
        format,
        { width: mipSize[0], height: mipSize[1], depth: mipSize[2] },
        actual
      );
      dst.destroy();
    });
  }

  testRun({
    textureDataLayout,
    copySize,
    dataSize,
    mipLevel = 0,
    origin = { x: 0, y: 0, z: 0 },
    textureSize,
    format,
    dimension = '2d',
    initMethod,
    checkMethod,
  }: {
    textureDataLayout: GPUTextureDataLayout;
    copySize: GPUExtent3DDict;
    dataSize: number;
    mipLevel?: number;
    origin?: Required<GPUOrigin3DDict>;
    textureSize: [number, number, number];
    format: GPUTextureFormat;
    dimension?: GPUTextureDimension;
    initMethod: string;
    checkMethod: string;
  }): void {
    const texture = this.device.createTexture({
      size: textureSize,
      format,
      dimension,
      mipLevelCount: mipLevel + 1,
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });

    const data = this.generateData(dataSize);

    switch (checkMethod) {
      case 'PartialCopyT2B': {
        this.initTexture(
          { texture, mipLevel, origin },
          textureDataLayout,
          copySize,
          data.slice(0, dataSize),
          initMethod
        );

        this.checkData({ texture, mipLevel, origin }, textureDataLayout, format, copySize, data);

        break;
      }
      case 'FullCopyT2B': {
        const fullTextureCopyLayout = getTextureCopyLayout(format, dimension, textureSize, {
          mipLevel,
        });

        const fullData = this.getFullData({ texture, mipLevel }, fullTextureCopyLayout);

        this.initTexture(
          { texture, mipLevel, origin },
          textureDataLayout,
          copySize,
          data.slice(0, dataSize),
          initMethod
        );

        this.fullCheck(
          { texture, mipLevel, origin },
          fullTextureCopyLayout,
          textureDataLayout,
          copySize,
          format,
          fullData,
          data
        );

        break;
      }
      default:
        unreachable();
    }
  }
}

// This is a helper function used for filtering test parameters
// TODO: Modify this after introducing tests with rendering.
function formatCanBeTested({ format }: { format: GPUTextureFormat }): boolean {
  return kTextureFormatInfo[format].copyDst && kTextureFormatInfo[format].copySrc;
}

const kAllInitMethods = ['WriteTexture', 'CopyB2T'] as const;
const kAllCheckMethods = ['PartialCopyT2B', 'FullCopyT2B'] as const;

export const g = makeTestGroup(CopyBetweenLinearDataAndTextureTest);

// Test that copying data with various bytesPerRow and rowsPerImage values and minimum required
// bytes in copy works for every format.
// Covers a special code path for Metal:
//    bufferSize - offset < bytesPerImage * copyExtent.depth
// Covers a special code path for D3D12:
//    when bytesPerRow is not a multiple of 512 and copyExtent.depth > 1: copyExtent.depth % 2 {==, >} 0
g.test('copy_with_various_rows_per_image_and_bytes_per_row')
  .params(
    params()
      .combine(poptions('initMethod', kAllInitMethods))
      .combine(poptions('checkMethod', kAllCheckMethods))
      .combine([
        { bytesPerRowPadding: 0, rowsPerImagePaddingInBlocks: 0 }, // no padding
        { bytesPerRowPadding: 0, rowsPerImagePaddingInBlocks: 6 }, // rowsPerImage padding
        { bytesPerRowPadding: 6, rowsPerImagePaddingInBlocks: 0 }, // bytesPerRow padding
        { bytesPerRowPadding: 15, rowsPerImagePaddingInBlocks: 17 }, // both paddings
      ])
      .combine([
        { copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 5 }, // standard copy, covers bytesPerRow = 256 and copyDepth % 2 == 1.
        { copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 2 }, // standard copy, covers bytesPerRow = 256 and copyDepth % 2 == 0.
        { copyWidthInBlocks: 256, copyHeightInBlocks: 3, copyDepth: 2 }, // copyWidth is 256-aligned
        { copyWidthInBlocks: 0, copyHeightInBlocks: 4, copyDepth: 5 }, // empty copy because of width
        { copyWidthInBlocks: 3, copyHeightInBlocks: 0, copyDepth: 5 }, // empty copy because of height
        { copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 0 }, // empty copy because of depth
        { copyWidthInBlocks: 1, copyHeightInBlocks: 4, copyDepth: 5 }, // copyWidth = 1
        { copyWidthInBlocks: 3, copyHeightInBlocks: 1, copyDepth: 5 }, // copyHeight = 1
        { copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 1 }, // copyDepth = 1
        { copyWidthInBlocks: 7, copyHeightInBlocks: 1, copyDepth: 1 }, // copyHeight = 1 and copyDepth = 1
      ])
      .combine(poptions('format', kTextureFormats))
      .filter(formatCanBeTested)
  )
  .fn(async t => {
    const {
      bytesPerRowPadding,
      rowsPerImagePaddingInBlocks,
      copyWidthInBlocks,
      copyHeightInBlocks,
      copyDepth,
      format,
      initMethod,
      checkMethod,
    } = t.params;

    const info = kTextureFormatInfo[format];

    // For CopyB2T and CopyT2B we need to have bytesPerRow 256-aligned,
    // to make this happen we align the bytesInACompleteRow value and multiply
    // bytesPerRowPadding by 256.
    const bytesPerRowAlignment =
      initMethod === 'WriteTexture' && checkMethod === 'FullCopyT2B' ? 1 : 256;

    const copyWidth = copyWidthInBlocks * info.blockWidth!;
    const copyHeight = copyHeightInBlocks * info.blockHeight!;
    const rowsPerImage = copyHeight + rowsPerImagePaddingInBlocks * info.blockHeight!;
    const bytesPerRow =
      align(t.bytesInACompleteRow(copyWidth, format), bytesPerRowAlignment) +
      bytesPerRowPadding * bytesPerRowAlignment;
    const copySize = { width: copyWidth, height: copyHeight, depth: copyDepth };

    const minDataSize = t.requiredBytesInCopy({ bytesPerRow, rowsPerImage }, format, copySize);

    t.testRun({
      textureDataLayout: { offset: 0, bytesPerRow, rowsPerImage },
      copySize,
      dataSize: minDataSize,
      textureSize: [copyWidth, copyHeight, copyDepth],
      format,
      initMethod,
      checkMethod,
    });
  });

// Test that copying data with various offset values and additional data paddings
// works for every format with 2d and 2d-array textures.
// Covers two special code paths for D3D12:
//   offset + bytesInCopyExtentPerRow { ==, > } bytesPerRow
//   offset > bytesInACompleteCopyImage
g.test('copy_with_various_offsets_and_data_sizes')
  .params(
    params()
      .combine(poptions('initMethod', kAllInitMethods))
      .combine(poptions('checkMethod', kAllCheckMethods))
      .combine([
        { offsetInBlocks: 0, dataPaddingInBytes: 0 }, // no offset and no padding
        { offsetInBlocks: 1, dataPaddingInBytes: 0 }, // offset = 1
        { offsetInBlocks: 2, dataPaddingInBytes: 0 }, // offset = 2
        { offsetInBlocks: 15, dataPaddingInBytes: 0 }, // offset = 15
        { offsetInBlocks: 16, dataPaddingInBytes: 0 }, // offset = 16
        { offsetInBlocks: 242, dataPaddingInBytes: 0 }, // for rgba8unorm format: offset + bytesInCopyExtentPerRow = 242 + 12 = 256 = bytesPerRow
        { offsetInBlocks: 243, dataPaddingInBytes: 0 }, // for rgba8unorm format: offset + bytesInCopyExtentPerRow = 243 + 12 > 256 = bytesPerRow
        { offsetInBlocks: 768, dataPaddingInBytes: 0 }, // for copyDepth = 1 and blockWidth = 1: offset = 768 = 3 * 256 = bytesInACompleteCopyImage
        { offsetInBlocks: 769, dataPaddingInBytes: 0 }, // for copyDepth = 1 and blockWidth = 1: offset = 769 > 768 = bytesInACompleteCopyImage
        { offsetInBlocks: 0, dataPaddingInBytes: 1 }, // dataPaddingInBytes > 0
        { offsetInBlocks: 1, dataPaddingInBytes: 8 }, // offset > 0 and dataPaddingInBytes > 0
      ])
      .combine(poptions('copyDepth', [1, 2])) // 2d and 2d-array textures
      .combine(poptions('format', kTextureFormats))
      .filter(formatCanBeTested)
  )
  .fn(async t => {
    const {
      offsetInBlocks,
      dataPaddingInBytes,
      copyDepth,
      format,
      initMethod,
      checkMethod,
    } = t.params;

    const info = kTextureFormatInfo[format];

    const offset = offsetInBlocks * info.bytesPerBlock!;
    const copySize = {
      width: 3 * info.blockWidth!,
      height: 3 * info.blockHeight!,
      depth: copyDepth,
    };
    const rowsPerImage = copySize.height;
    const bytesPerRow = 256;

    const dataSize =
      offset +
      t.requiredBytesInCopy({ offset, bytesPerRow, rowsPerImage }, format, copySize) +
      dataPaddingInBytes;

    // We're copying a (3 x 3 x copyDepth) (in texel blocks) part of a (4 x 4 x copyDepth) (in texel blocks)
    // texture with no origin.
    t.testRun({
      textureDataLayout: { offset, bytesPerRow, rowsPerImage },
      copySize,
      dataSize,
      textureSize: [4 * info.blockWidth!, 4 * info.blockHeight!, copyDepth],
      format,
      initMethod,
      checkMethod,
    });
  });

// Test that copying slices of a texture works with various origin and copyExtent values
// for all formats.
g.test('copy_with_various_origins_and_copy_extents')
  .params(
    params()
      .combine(poptions('initMethod', kAllInitMethods))
      .combine(poptions('checkMethod', kAllCheckMethods))
      .combine(poptions('originValueInBlocks', [0, 7, 8]))
      .combine(poptions('copySizeValueInBlocks', [0, 7, 8]))
      .combine(poptions('textureSizePaddingValueInBlocks', [0, 7, 8]))
      .unless(p => {
        return (
          p.copySizeValueInBlocks + p.originValueInBlocks + p.textureSizePaddingValueInBlocks === 0
        );
      }) // we can't create an empty texture
      .combine(poptions('coordinateToTest', ['width', 'height', 'depth'] as const))
      .combine(poptions('format', kTextureFormats))
      .filter(formatCanBeTested)
  )
  .fn(async t => {
    const {
      coordinateToTest,
      originValueInBlocks,
      copySizeValueInBlocks,
      textureSizePaddingValueInBlocks,
      format,
      initMethod,
      checkMethod,
    } = t.params;

    const info = kTextureFormatInfo[format];

    const origin = { x: info.blockWidth!, y: info.blockHeight!, z: 1 };
    const copySize = { width: 2 * info.blockWidth!, height: 2 * info.blockHeight!, depth: 2 };
    const textureSize: [number, number, number] = [3 * info.blockWidth!, 3 * info.blockHeight!, 3];

    switch (coordinateToTest) {
      case 'width': {
        origin.x = originValueInBlocks * info.blockWidth!;
        copySize.width = copySizeValueInBlocks * info.blockWidth!;
        textureSize[0] =
          origin.x + copySize.width + textureSizePaddingValueInBlocks * info.blockWidth!;
        break;
      }
      case 'height': {
        origin.y = originValueInBlocks * info.blockHeight!;
        copySize.height = copySizeValueInBlocks * info.blockHeight!;
        textureSize[1] =
          origin.y + copySize.height + textureSizePaddingValueInBlocks * info.blockHeight!;
        break;
      }
      case 'depth': {
        origin.z = originValueInBlocks;
        copySize.depth = copySizeValueInBlocks;
        textureSize[2] = origin.z + copySize.depth + textureSizePaddingValueInBlocks;
        break;
      }
    }

    const rowsPerImage = copySize.height;
    const bytesPerRow = align(copySize.width, 256);
    const dataSize = t.requiredBytesInCopy({ bytesPerRow, rowsPerImage }, format, copySize);

    // For testing width: we copy a (_ x 2 x 2) (in texel blocks) part of a (_ x 3 x 3) (in texel blocks)
    // texture with origin (_, 1, 1) (in texel blocks). Similarly for other coordinates.
    t.testRun({
      textureDataLayout: { offset: 0, bytesPerRow, rowsPerImage },
      copySize,
      dataSize,
      origin,
      textureSize,
      format,
      initMethod,
      checkMethod,
    });
  });

function* textureSizeExpander({
  format,
  mipLevel,
  texturePhysicalSizeAtMipLevelInBlocks,
}: {
  format: GPUTextureFormat;
  mipLevel: number;
  texturePhysicalSizeAtMipLevelInBlocks: GPUExtent3DDict;
}): Generator<{ textureSize: [number, number, number] }> {
  const info = kTextureFormatInfo[format];
  const textureSize = [
    (texturePhysicalSizeAtMipLevelInBlocks.width * info.blockWidth!) << mipLevel,
    (texturePhysicalSizeAtMipLevelInBlocks.height * info.blockHeight!) << mipLevel,
    texturePhysicalSizeAtMipLevelInBlocks.depth,
  ];
  yield {
    textureSize: textureSize as [number, number, number],
  };
  if (info.blockWidth! > 1 && mipLevel > 0) {
    yield {
      textureSize: [textureSize[0] - (1 << mipLevel), textureSize[1], textureSize[2]],
    };
  }
  if (info.blockHeight! > 1 && mipLevel > 0) {
    yield {
      textureSize: [textureSize[0], textureSize[1] - (1 << mipLevel), textureSize[2]],
    };
  }
  if (info.blockHeight! > 1 && info.blockWidth! > 1 && mipLevel > 0) {
    yield {
      textureSize: [
        textureSize[0] - (1 << mipLevel),
        textureSize[1] - (1 << mipLevel),
        textureSize[2],
      ],
    };
  }
}

// Test that copying various mip levels works.
g.test('copy_various_mip_levels')
  .params(
    params()
      .combine(poptions('initMethod', kAllInitMethods))
      .combine(poptions('checkMethod', kAllCheckMethods))
      .combine([
        {
          copySizeInBlocks: { width: 2, height: 2, depth: 2 },
          originInBlocks: { x: 3, y: 2, z: 1 },
          texturePhysicalSizeAtMipLevelInBlocks: { width: 6, height: 5, depth: 4 },
          mipLevel: 1,
        }, // copying on mipLevel 1
        {
          copySizeInBlocks: { width: 5, height: 6, depth: 1 },
          originInBlocks: { x: 1, y: 1, z: 0 },
          texturePhysicalSizeAtMipLevelInBlocks: { width: 6, height: 7, depth: 1 },
          mipLevel: 5,
        }, // copying on mipLevel 5 of a 2d texture
        {
          copySizeInBlocks: { width: 5, height: 6, depth: 2 },
          originInBlocks: { x: 1, y: 1, z: 1 },
          texturePhysicalSizeAtMipLevelInBlocks: { width: 6, height: 7, depth: 3 },
          mipLevel: 5,
        }, // copying on mipLevel 5 of a 2d-array texture
      ])
      .combine(poptions('format', kTextureFormats))
      .filter(formatCanBeTested)
      .expand(textureSizeExpander)
  )
  .fn(async t => {
    const {
      copySizeInBlocks,
      originInBlocks,
      textureSize,
      mipLevel,
      format,
      initMethod,
      checkMethod,
    } = t.params;

    const info = kTextureFormatInfo[format];

    const origin = {
      x: originInBlocks.x * info.blockWidth!,
      y: originInBlocks.y * info.blockHeight!,
      z: originInBlocks.z,
    };
    const copySize = {
      width: copySizeInBlocks.width * info.blockWidth!,
      height: copySizeInBlocks.height * info.blockHeight!,
      depth: copySizeInBlocks.depth,
    };

    const rowsPerImage = copySize.height;
    const bytesPerRow = align(copySize.width, 256);
    const dataSize = t.requiredBytesInCopy({ bytesPerRow, rowsPerImage }, format, copySize);

    t.testRun({
      textureDataLayout: { offset: 0, bytesPerRow, rowsPerImage },
      copySize,
      dataSize,
      origin,
      mipLevel,
      textureSize,
      format,
      initMethod,
      checkMethod,
    });
  });
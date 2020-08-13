export const description = `writeTexture + copyBufferToTexture + copyTextureToBuffer operation tests.
* TODO: add another initMethod which renders the texture
* TODO: check that CopyT2B doesn't overwrite any other bytes
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

    assert(texel.x >= origin.x && texel.y >= origin.y && texel.z >= origin.z);
    assert(rowsPerImage! % kTextureFormatInfo[format].blockHeight! === 0);
    assert(texel.x % kTextureFormatInfo[format].blockWidth! === 0);
    assert(texel.y % kTextureFormatInfo[format].blockHeight! === 0);
    assert(origin.x % kTextureFormatInfo[format].blockWidth! === 0);
    assert(origin.y % kTextureFormatInfo[format].blockHeight! === 0);

    const bytesPerImage = (rowsPerImage! / kTextureFormatInfo[format].blockHeight!) * bytesPerRow;

    return (
      offset! +
      (texel.z - origin.z) * bytesPerImage +
      ((texel.y - origin.y) / kTextureFormatInfo[format].blockHeight!) * bytesPerRow +
      ((texel.x - origin.x) / kTextureFormatInfo[format].blockWidth!) *
        kTextureFormatInfo[format].bytesPerBlock!
    );
  }

  *iterateBlockRows(
    size: GPUExtent3DDict,
    origin: Required<GPUOrigin3DDict>,
    format: GPUTextureFormat
  ): Generator<Required<GPUOrigin3DDict>> {
    assert(size.height % kTextureFormatInfo[format].blockHeight! === 0);
    for (let y = 0; y < size.height / kTextureFormatInfo[format].blockHeight!; ++y) {
      for (let z = 0; z < size.depth; ++z) {
        yield {
          x: origin.x,
          y: origin.y + y * kTextureFormatInfo[format].blockHeight!,
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
      size: byteLength + 3, // the additional 3 bytes are needed for expectContents
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
    // other eventual async expectations that it will be correct.
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
    origin: Required<GPUOrigin3DDict>;
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
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });

    // The additional 3 bytes are needed for expectContents. We don't pass them to initTexture.
    const data = this.generateData(dataSize + 3);

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
function formatCopyableWithMethod({ format }: { format: GPUTextureFormat }): boolean {
  return kTextureFormatInfo[format].copyDst && kTextureFormatInfo[format].copySrc;
}

const kAllInitMethods = ['WriteTexture', 'CopyB2T'] as const;
const kAllCheckMethods = ['PartialCopyT2B', 'FullCopyT2B'] as const;

export const g = makeTestGroup(CopyBetweenLinearDataAndTextureTest);

g.test('copy_with_various_data_paddings')
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
        { copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // standard copy
        { copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 3, offsetInBlocks: 11 }, // standard copy, offset > 0
        { copyWidthInBlocks: 256, copyHeightInBlocks: 3, copyDepth: 2, offsetInBlocks: 0 }, // copyWidth is 256-aligned
        { copyWidthInBlocks: 0, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // empty copy because of width
        { copyWidthInBlocks: 3, copyHeightInBlocks: 0, copyDepth: 5, offsetInBlocks: 0 }, // empty copy because of height
        { copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 0, offsetInBlocks: 13 }, // empty copy because of depth, offset > 0
        { copyWidthInBlocks: 1, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // copyWidth = 1
        { copyWidthInBlocks: 3, copyHeightInBlocks: 1, copyDepth: 5, offsetInBlocks: 15 }, // copyHeight = 1, offset > 0
        { copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 1, offsetInBlocks: 0 }, // copyDepth = 1
        { copyWidthInBlocks: 7, copyHeightInBlocks: 1, copyDepth: 1, offsetInBlocks: 0 }, // copyHeight = 1 and copyDepth = 1
      ])
      .combine(poptions('format', kTextureFormats))
      .filter(formatCopyableWithMethod)
  )
  .fn(async t => {
    const {
      offsetInBlocks,
      bytesPerRowPadding,
      rowsPerImagePaddingInBlocks,
      copyWidthInBlocks,
      copyHeightInBlocks,
      copyDepth,
      format,
      initMethod,
      checkMethod,
    } = t.params;

    // For CopyB2T and CopyT2B we need to have bytesPerRow 256-aligned,
    // to make this happen we align the bytesInACompleteRow value and multiply
    // bytesPerRowPadding by 256.
    const bytesPerRowAlignment =
      initMethod === 'WriteTexture' && checkMethod === 'FullCopyT2B' ? 1 : 256;

    const copyWidth = copyWidthInBlocks * kTextureFormatInfo[format].blockWidth!;
    const copyHeight = copyHeightInBlocks * kTextureFormatInfo[format].blockHeight!;
    const offset = offsetInBlocks * kTextureFormatInfo[format].bytesPerBlock!;
    const rowsPerImage =
      copyHeight + rowsPerImagePaddingInBlocks * kTextureFormatInfo[format].blockHeight!;
    const bytesPerRow =
      align(t.bytesInACompleteRow(copyWidth, format), bytesPerRowAlignment) +
      bytesPerRowPadding * bytesPerRowAlignment;
    const copySize = { width: copyWidth, height: copyHeight, depth: copyDepth };

    const minDataSize =
      offset + t.requiredBytesInCopy({ offset, bytesPerRow, rowsPerImage }, format, copySize);

    t.testRun({
      textureDataLayout: { offset, bytesPerRow, rowsPerImage },
      copySize,
      dataSize: minDataSize,
      origin: {
        x: kTextureFormatInfo[format].blockWidth!,
        y: kTextureFormatInfo[format].blockHeight!,
        z: 1,
      },
      textureSize: [
        copyWidth + 2 * kTextureFormatInfo[format].blockWidth!,
        copyHeight + 2 * kTextureFormatInfo[format].blockHeight!,
        copyDepth + 2,
      ],
      format,
      initMethod,
      checkMethod,
    });
  });

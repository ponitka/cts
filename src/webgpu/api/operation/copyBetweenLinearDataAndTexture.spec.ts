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

// Offset for a particular texel in the linear texture data
function getTexelOffsetInBytes(
  textureDataLayout: GPUTextureDataLayout,
  format: GPUTextureFormat,
  texel: Required<GPUOrigin3DDict>,
  origin: Required<GPUOrigin3DDict> = { x: 0, y: 0, z: 0 }
): number {
  assert(texel.x >= origin.x && texel.y >= origin.y && texel.z >= origin.z);
  assert((texel.x - origin.x) % kTextureFormatInfo[format].blockWidth! === 0);
  assert((texel.y - origin.y) % kTextureFormatInfo[format].blockHeight! === 0);

  const { offset, bytesPerRow, rowsPerImage } = textureDataLayout;
  const bytesPerImage = (rowsPerImage! / kTextureFormatInfo[format].blockHeight!) * bytesPerRow;

  return (
    offset! +
    (texel.z - origin.z) * bytesPerImage +
    ((texel.y - origin.y) / kTextureFormatInfo[format].blockHeight!) * bytesPerRow +
    ((texel.x - origin.x) / kTextureFormatInfo[format].blockWidth!) *
      kTextureFormatInfo[format].bytesPerBlock!
  );
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

  generateData(byteSize: number): Uint8Array {
    const arr = new Uint8Array(byteSize);
    for (let i = 0; i < byteSize; ++i) {
      arr[i] = i % 251;
    }
    return arr;
  }

  // Put data into texture with an appropriate method.
  initTexture(
    textureCopyView: GPUTextureCopyView,
    textureDataLayout: GPUTextureDataLayout,
    size: GPUExtent3D,
    data: Uint8Array,
    method: string
  ): void {
    switch (method) {
      case 'WriteTexture': {
        this.device.defaultQueue.writeTexture(textureCopyView, data, textureDataLayout, size);

        break;
      }
      case 'CopyB2T': {
        const buffer = this.device.createBuffer({
          mappedAtCreation: true,
          size: data.byteLength,
          usage: GPUBufferUsage.COPY_SRC,
        });
        new Uint8Array(buffer.getMappedRange()).set(data);
        buffer.unmap();

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToTexture({ buffer, ...textureDataLayout }, textureCopyView, size);
        this.device.defaultQueue.submit([encoder.finish()]);

        break;
      }
      default:
        unreachable();
    }
  }

  check(
    textureCopyView: GPUTextureCopyView,
    origin: Required<GPUOrigin3DDict>,
    textureDataLayout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    size: GPUExtent3DDict,
    expected: Uint8Array
  ): void {
    const buffer = this.device.createBuffer({
      size: expected.byteLength + 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(textureCopyView, { buffer, ...textureDataLayout }, size);
    this.device.defaultQueue.submit([encoder.finish()]);

    for (let y = 0; y < size.height / kTextureFormatInfo[format].blockHeight!; ++y) {
      for (let z = 0; z < size.depth; ++z) {
        const texel = {
          x: origin.x,
          y: origin.y + y * kTextureFormatInfo[format].blockHeight!,
          z: origin.z + z,
        };
        const rowOffset = getTexelOffsetInBytes(textureDataLayout, format, texel, origin);
        const rowLength =
          (size.width / kTextureFormatInfo[format].blockWidth!) *
          kTextureFormatInfo[format].bytesPerBlock!;
        this.expectContents(buffer, expected.slice(rowOffset, rowOffset + rowLength), rowOffset);
      }
    }
  }

  getFullData(
    textureCopyView: GPUTextureCopyView,
    fullTextureCopyLayout: TextureCopyLayout
  ): GPUBuffer {
    const { texture, mipLevel } = textureCopyView;
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
    format: GPUTextureFormat,
    textureDataLayout: GPUTextureDataLayout,
    fullTextureCopyLayout: TextureCopyLayout,
    size: GPUExtent3DDict,
    origin: Required<GPUOrigin3DDict>,
    fullData: Uint8Array,
    partialData: Uint8Array
  ): void {
    const { bytesPerRow, rowsPerImage } = fullTextureCopyLayout;

    for (let y = 0; y < size.height / kTextureFormatInfo[format].blockHeight!; ++y) {
      for (let z = 0; z < size.depth; ++z) {
        const texel = {
          x: origin.x,
          y: origin.y + y * kTextureFormatInfo[format].blockHeight!,
          z: origin.z + z,
        };
        const partialDataOffset = getTexelOffsetInBytes(textureDataLayout, format, texel, origin);
        const fullDataOffset = getTexelOffsetInBytes(
          { bytesPerRow, rowsPerImage, offset: 0 },
          format,
          texel
        );
        const rowLength =
          (size.width / kTextureFormatInfo[format].blockWidth!) *
          kTextureFormatInfo[format].bytesPerBlock!;
        for (let b = 0; b < rowLength; ++b) {
          fullData[fullDataOffset + b] = partialData[partialDataOffset + b];
        }
      }
    }
  }

  fullCheck(
    fullTextureCopyLayout: TextureCopyLayout,
    textureCopyView: GPUTextureCopyView,
    origin: Required<GPUOrigin3DDict>,
    textureDataLayout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    fullData: GPUBuffer,
    partialData: Uint8Array
  ): void {
    const { texture, mipLevel } = textureCopyView;
    const { mipSize, bytesPerRow, rowsPerImage, byteLength } = fullTextureCopyLayout;
    const { dst, begin, end } = this.createAlignedCopyForMapRead(fullData, byteLength, 0);

    this.eventualAsyncExpectation(async _ => {
      await dst.mapAsync(GPUMapMode.READ);
      const actual = new Uint8Array(dst.getMappedRange()).slice(begin, end);
      this.updateFullData(
        format,
        textureDataLayout,
        fullTextureCopyLayout,
        { width: mipSize[0], height: mipSize[1], depth: mipSize[2] },
        origin,
        actual,
        partialData
      );
      this.check(
        { texture, mipLevel },
        { x: 0, y: 0, z: 0 },
        { bytesPerRow, rowsPerImage, offset: 0 },
        format,
        { width: mipSize[0], height: mipSize[1], depth: mipSize[2] },
        actual
      );
      dst.destroy();
    });
  }

  testRun(
    textureCopyView: GPUTextureCopyView,
    textureDesc: GPUTextureDescriptor,
    textureDataLayout: GPUTextureDataLayout,
    size: GPUExtent3DDict,
    {
      dataSize,
      origin = { x: 0, y: 0, z: 0 },
      initMethod,
      checkMethod,
    }: {
      dataSize: number;
      origin?: Required<GPUOrigin3DDict>;
      initMethod: string;
      checkMethod: string;
    }
  ): void {
    const data = this.generateData(align(dataSize, 4) + 4);

    switch (checkMethod) {
      case 'PartialCopyT2B': {
        this.initTexture(textureCopyView, textureDataLayout, size, data, initMethod);

        this.check(textureCopyView, origin, textureDataLayout, textureDesc.format, size, data);

        break;
      }
      case 'FullCopyT2B': {
        const fullTextureCopyLayout = getTextureCopyLayout(
          textureDesc.format,
          textureDesc.dimension!,
          [size.width, size.height, size.depth],
          { mipLevel: textureCopyView.mipLevel! }
        );

        const fullData = this.getFullData(textureCopyView, fullTextureCopyLayout);

        this.initTexture(textureCopyView, textureDataLayout, size, data, initMethod);

        this.fullCheck(
          fullTextureCopyLayout,
          textureCopyView,
          origin,
          textureDataLayout,
          textureDesc.format,
          fullData,
          data
        );

        break;
      }
      default:
        unreachable();
    }
  }

  // This is a helper function used for creating a texture when we don't have to be very
  // precise about its size as long as it's big enough and properly aligned.
  getAlignedTextureDescriptor(
    format: GPUTextureFormat,
    copySize: GPUExtent3DDict = { width: 1, height: 1, depth: 1 },
    origin: Required<GPUOrigin3DDict> = { x: 0, y: 0, z: 0 }
  ): GPUTextureDescriptor {
    return {
      size: {
        width: Math.max(1, copySize.width + origin.x) * kTextureFormatInfo[format].blockWidth!,
        height: Math.max(1, copySize.height + origin.y) * kTextureFormatInfo[format].blockHeight!,
        depth: Math.max(1, copySize.depth + origin.z),
      },
      dimension: '2d',
      format,
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    };
  }
}

interface WithFormat {
  format: GPUTextureFormat;
}

// This is a helper function used for filtering test parameters
function formatCopyableWithMethod({ format }: WithFormat): boolean {
  return kTextureFormatInfo[format].copyable && !kTextureFormatInfo[format].depth;
}

const kAllInitMethods = ['WriteTexture', 'CopyB2T'] as const;
const kAllCheckMethods = ['PartialCopyT2B', 'FullCopyT2B'] as const;

export const g = makeTestGroup(CopyBetweenLinearDataAndTextureTest);

g.test('copy_whole_texture')
  .params(
    params()
      .combine(poptions('initMethod', kAllInitMethods))
      .combine(poptions('checkMethod', kAllCheckMethods))
      .combine(poptions('width', [1, 15, 16]))
      .combine(poptions('height', [1, 7, 8]))
      .combine(poptions('depth', [1, 3, 4]))
  )
  .fn(async t => {
    const { initMethod, checkMethod, width, height, depth } = t.params;

    const textureDesc: GPUTextureDescriptor = {
      size: { width, height, depth },
      format: 'rgba8unorm',
      dimension: '2d',
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    };
    const texture = t.device.createTexture(textureDesc);
    const textureCopyView = { texture };
    const textureDataLayout = { offset: 0, bytesPerRow: 256, rowsPerImage: height };
    const copyExtent = { width, height, depth };

    const dataSize = t.requiredBytesInCopy(textureDataLayout, textureDesc.format, copyExtent);

    t.testRun(textureCopyView, textureDesc, textureDataLayout, copyExtent, {
      dataSize,
      initMethod,
      checkMethod,
    });
  });

g.test('copy_with_data_paddings')
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
        { copyWidthInBlocks: 2, copyHeightInBlocks: 3, copyDepth: 3, offsetInBlocks: 0 },
        { copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // standard copy
        { copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 3, offsetInBlocks: 11 }, // standard copy, offset > 0
        { copyWidthInBlocks: 256, copyHeightInBlocks: 3, copyDepth: 2, offsetInBlocks: 0 }, // copyWidth is 256-aligned
        //{ copyWidthInBlocks: 0, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // empty copy because of width
        //{ copyWidthInBlocks: 3, copyHeightInBlocks: 0, copyDepth: 5, offsetInBlocks: 0 }, // empty copy because of height
        //{/ copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 0, offsetInBlocks: 13 }, // empty copy because of depth, offset > 0
        //{ copyWidthInBlocks: 1, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // copyWidth = 1
        //{ copyWidthInBlocks: 3, copyHeightInBlocks: 1, copyDepth: 5, offsetInBlocks: 15 }, // copyHeight = 1, offset > 0
        //{ copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 1, offsetInBlocks: 0 }, // copyDepth = 1
        //{ copyWidthInBlocks: 7, copyHeightInBlocks: 1, copyDepth: 1, offsetInBlocks: 0 }, // copyHeight = 1 and copyDepth = 1
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
    const size = { width: copyWidth, height: copyHeight, depth: copyDepth };

    const minDataSize =
      offset + t.requiredBytesInCopy({ offset, bytesPerRow, rowsPerImage }, format, size);

    const textureDesc = t.getAlignedTextureDescriptor(format, size);
    const texture = t.device.createTexture(textureDesc);

    t.testRun({ texture }, textureDesc, { offset, bytesPerRow, rowsPerImage }, size, {
      dataSize: minDataSize,
      initMethod,
      checkMethod,
    });
  });

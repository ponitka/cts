export const description =
  'writeTexture + copyBufferToTexture + copyTextureToBuffer operation tests.';

import { params, poptions } from '../../../common/framework/params_builder.js';
import { makeTestGroup } from '../../../common/framework/test_group.js';
import { assert, unreachable } from '../../../common/framework/util/util.js';
import { kTextureFormatInfo, kTextureFormats } from '../../capability_info.js';
import { GPUTest } from '../../gpu_test.js';
import { align } from '../../util/math.js';
import { getTextureCopyLayout, TextureCopyLayout } from '../../util/texture/layout.js';

// Offset for a particular texel block in the linear texture data
function getTexelOffsetInBytes(
  textureDataLayout: GPUTextureDataLayout,
  format: GPUTextureFormat,
  texel: Required<GPUOrigin3DDict>, // coordinates of the first texel in the texel block
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

function getTexeBlockIndex(
  format: GPUTextureFormat,
  texel: Required<GPUOrigin3DDict>, // coordinates of the first texel in the texel block
  size: GPUExtent3DDict
): number {
  assert(texel.x % kTextureFormatInfo[format].blockWidth! === 0);
  assert(texel.y % kTextureFormatInfo[format].blockHeight! === 0);
  assert(size.width % kTextureFormatInfo[format].blockWidth! === 0);
  assert(size.height % kTextureFormatInfo[format].blockHeight! === 0);

  return (
    texel.x / kTextureFormatInfo[format].blockWidth! +
    (texel.y / kTextureFormatInfo[format].blockHeight!) *
      (size.width / kTextureFormatInfo[format].blockWidth!) +
    texel.z *
      (size.height / kTextureFormatInfo[format].blockHeight!) *
      (size.width * kTextureFormatInfo[format].blockWidth!)
  );
}

class FullTextureData {
  texelBlocks: Array<GPUBuffer> = new Array<GPUBuffer>(50);

  constructor(
    textureCopyView: GPUTextureCopyView,
    format: GPUTextureFormat,
    fullTextureCopyLayout: TextureCopyLayout,
    device: GPUDevice
  ) {
    const { mipSize } = fullTextureCopyLayout;
    const { mipLevel, texture } = textureCopyView;
    const bytesPerBlock = kTextureFormatInfo[format].bytesPerBlock!;

    for (let x = 0; x < mipSize[0] / kTextureFormatInfo[format].blockWidth!; ++x) {
      for (let y = 0; y < mipSize[1] / kTextureFormatInfo[format].blockHeight!; ++y) {
        for (let z = 0; z < mipSize[2]; ++z) {
          const buffer = device.createBuffer({
            size: align(bytesPerBlock, 4),
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });

          const texel = {
            x: x * kTextureFormatInfo[format].blockWidth!,
            y: y * kTextureFormatInfo[format].blockHeight!,
            z,
          };

          const encoder = device.createCommandEncoder();
          encoder.copyTextureToBuffer(
            {
              mipLevel,
              texture,
              origin: texel,
            },
            { buffer, bytesPerRow: 0 },
            {
              width: kTextureFormatInfo[format].blockWidth!,
              height: kTextureFormatInfo[format].blockHeight!,
              depth: 1,
            }
          );
          device.defaultQueue.submit([encoder.finish()]);

          this.texelBlocks[
            getTexeBlockIndex(format, texel, {
              width: mipSize[0],
              height: mipSize[1],
              depth: mipSize[2],
            })
          ] = buffer;
        }
      }
    }
  }

  update(
    format: GPUTextureFormat,
    textureDataLayout: GPUTextureDataLayout,
    size: GPUExtent3DDict,
    origin: Required<GPUOrigin3DDict>,
    data: Uint8Array,
    device: GPUDevice
  ): void {
    const bytesPerBlock = kTextureFormatInfo[format].bytesPerBlock!;

    for (let x = 0; x < size.width / kTextureFormatInfo[format].blockWidth!; ++x) {
      for (let y = 0; y < size.height / kTextureFormatInfo[format].blockHeight!; ++y) {
        for (let z = 0; z < size.depth; ++z) {
          const texel = {
            x: origin.x + x * kTextureFormatInfo[format].blockWidth!,
            y: origin.y + y * kTextureFormatInfo[format].blockHeight!,
            z: origin.z + z,
          };
          const dataOffset = getTexelOffsetInBytes(textureDataLayout, format, texel, origin);
          const blockIndex = getTexeBlockIndex(format, texel, size);
          device.defaultQueue.writeBuffer(
            this.texelBlocks[blockIndex],
            0,
            data,
            dataOffset,
            align(bytesPerBlock, 4)
          );
        }
      }
    }
  }
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

  // Copy data into texture with an appropriate method.
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

  partialCheck(
    textureCopyView: GPUTextureCopyView,
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

    // We check the data row by row.
    for (let y = 0; y < size.height / kTextureFormatInfo[format].blockHeight!; ++y) {
      for (let z = 0; z < size.depth; ++z) {
        const texel = { x: 0, y: y * kTextureFormatInfo[format].blockHeight!, z };
        const rowOffset = getTexelOffsetInBytes(textureDataLayout, format, texel);
        const rowLength =
          (size.width / kTextureFormatInfo[format].blockWidth!) *
          kTextureFormatInfo[format].bytesPerBlock!;
        this.expectContents(buffer, expected.slice(rowOffset, rowOffset + rowLength), rowOffset);
      }
    }
  }

  fullCheck(
    fullTextureCopyLayout: TextureCopyLayout,
    textureCopyView: GPUTextureCopyView,
    format: GPUTextureFormat,
    expected: FullTextureData
  ): void {
    const { mipSize, bytesPerRow, rowsPerImage, byteLength } = fullTextureCopyLayout;
    const size = { width: mipSize[0], height: mipSize[1], depth: mipSize[2] };

    const buffer = this.device.createBuffer({
      size: byteLength + 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(textureCopyView, { buffer, bytesPerRow, rowsPerImage }, size);
    this.device.defaultQueue.submit([encoder.finish()]);

    for (let x = 0; x < mipSize[0] / kTextureFormatInfo[format].blockWidth!; ++x) {
      for (let y = 0; y < mipSize[1] / kTextureFormatInfo[format].blockHeight!; ++y) {
        for (let z = 0; z < mipSize[2]; ++z) {
          const texel = {
            x: x * kTextureFormatInfo[format].blockWidth!,
            y: y * kTextureFormatInfo[format].blockHeight!,
            z,
          };
          const texelOffset = getTexelOffsetInBytes({ bytesPerRow, rowsPerImage }, format, texel);
          const blockIndex = getTexeBlockIndex(format, texel, size);
          this.expectEqualBuffers(
            buffer,
            texelOffset,
            expected.texelBlocks[blockIndex],
            0,
            kTextureFormatInfo[format].bytesPerBlock!
          );
        }
      }
    }
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
    const data = this.generateData(dataSize);

    switch (checkMethod) {
      case 'PartialCopyT2B': {
        this.initTexture(textureCopyView, textureDataLayout, size, data, initMethod);

        this.partialCheck(textureCopyView, textureDataLayout, textureDesc.format, size, data);

        break;
      }
      case 'FullCopyT2B': {
        const fullTextureCopyLayout = getTextureCopyLayout(
          textureDesc.format,
          textureDesc.dimension!,
          [size.width, size.height, size.depth],
          { mipLevel: textureCopyView.mipLevel! }
        );

        const fullData = new FullTextureData(
          textureCopyView,
          textureDesc.format,
          fullTextureCopyLayout,
          this.device
        );

        this.initTexture(textureCopyView, textureDataLayout, size, data, initMethod);

        fullData.update(textureDesc.format, textureDataLayout, size, origin, data, this.device);

        this.fullCheck(fullTextureCopyLayout, textureCopyView, textureDesc.format, fullData);

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
      .combine(poptions('format', kTextureFormats))
      .filter(formatCopyableWithMethod)
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
        //{ copyWidthInBlocks: 0, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // empty copy because of width
        //{ copyWidthInBlocks: 3, copyHeightInBlocks: 0, copyDepth: 5, offsetInBlocks: 0 }, // empty copy because of height
        //{/ copyWidthInBlocks: 3, copyHeightInBlocks: 4, copyDepth: 0, offsetInBlocks: 13 }, // empty copy because of depth, offset > 0
        //{ copyWidthInBlocks: 1, copyHeightInBlocks: 4, copyDepth: 5, offsetInBlocks: 0 }, // copyWidth = 1
        //{ copyWidthInBlocks: 3, copyHeightInBlocks: 1, copyDepth: 5, offsetInBlocks: 15 }, // copyHeight = 1, offset > 0
        //{ copyWidthInBlocks: 5, copyHeightInBlocks: 4, copyDepth: 1, offsetInBlocks: 0 }, // copyDepth = 1
        //{ copyWidthInBlocks: 7, copyHeightInBlocks: 1, copyDepth: 1, offsetInBlocks: 0 }, // copyHeight = 1 and copyDepth = 1
      ])
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

export const description =
  'writeTexture + copyBufferToTexture + copyTextureToBuffer operation tests.';

import { params, poptions } from '../../../common/framework/params_builder.js';
import { makeTestGroup } from '../../../common/framework/test_group.js';
import { assert, unreachable } from '../../../common/framework/util/util.js';
import { kTextureFormatInfo } from '../../capability_info.js';
import { GPUTest } from '../../gpu_test.js';
import { getTextureCopyLayout, TextureCopyLayout } from '../../util/texture/layout.js';

export class CopyBetweenLinearDataAndTextureTest extends GPUTest {
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

  // This sets all the bytes that aren't involved in the texture copy to 0.
  fillDataWithZeros(
    textureDataLayout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    size: GPUExtent3DDict,
    data: Uint8Array
  ): void {
    const { bytesPerRow, rowsPerImage, offset } = textureDataLayout;
    const bytesPerImage = (rowsPerImage! / kTextureFormatInfo[format].blockHeight!) * bytesPerRow;

    for (let b = 0; b < data.byteLength; ++b) {
      // (x, y, z) are the coordinates of the left-bottom pixel of the texel to which the b-th byte would correspond.
      const z = Math.floor((b - offset!) / bytesPerImage);
      const y =
        Math.floor(((b - offset!) % bytesPerImage) / bytesPerRow) *
        kTextureFormatInfo[format].blockHeight!;
      const x =
        Math.floor(((b - offset!) % bytesPerRow) / kTextureFormatInfo[format].bytesPerBlock!) *
        kTextureFormatInfo[format].blockWidth!;

      if (b < offset! || x >= size.width || y >= size.height || z >= size.depth) {
        data[b] = 0;
      }
    }
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

  // Stores data for the whole texture, used in fullCheck.
  getFullData(
    textureCopyView: GPUTextureCopyView,
    fullTextureCopyLayout: TextureCopyLayout
  ): GPUBuffer {
    const { bytesPerRow, rowsPerImage, byteLength, mipSize } = fullTextureCopyLayout;
    const { mipLevel, texture } = textureCopyView;

    const buffer = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { mipLevel, texture },
      { buffer, bytesPerRow, rowsPerImage },
      mipSize
    );
    this.device.defaultQueue.submit([encoder.finish()]);

    return buffer;
  }

  // Offset for a particular texel block in the linear texture data
  getTexelOffset(
    textureDataLayout: GPUTextureDataLayout,
    format: GPUTextureFormat,
    point: Required<GPUOrigin3DDict> // coordinates relative to the texture subresource
  ): number {
    assert(point.x % kTextureFormatInfo[format].blockWidth! === 0);
    assert(point.y % kTextureFormatInfo[format].blockHeight! === 0);

    const { offset, bytesPerRow, rowsPerImage } = textureDataLayout;
    const bytesPerImage = (rowsPerImage! / kTextureFormatInfo[format].blockHeight!) * bytesPerRow;

    return (
      offset! +
      point.z * bytesPerImage +
      (point.y / kTextureFormatInfo[format].blockHeight!) * bytesPerRow +
      (point.x / kTextureFormatInfo[format].blockWidth!) * kTextureFormatInfo[format].bytesPerBlock!
    );
  }

  // Updates full texture data after a copy operation
  updateFullData(
    format: GPUTextureFormat,
    fullTextureCopyLayout: TextureCopyLayout,
    textureDataLayout: GPUTextureDataLayout,
    size: GPUExtent3DDict,
    origin: Required<GPUOrigin3DDict>,
    partialData: Uint8Array,
    fullData: GPUBuffer
  ): void {
    assert(size.width % kTextureFormatInfo[format].blockWidth! === 0);
    assert(size.height % kTextureFormatInfo[format].blockHeight! === 0);

    const { bytesPerRow, rowsPerImage } = fullTextureCopyLayout;

    // We copy the partial data into the full data row by row.
    for (let y = 0; y < size.height / kTextureFormatInfo[format].blockHeight!; ++y) {
      for (let z = 0; z < size.depth; ++z) {
        const point = {
          x: origin.x,
          y: origin.y + y * kTextureFormatInfo[format].blockHeight!,
          z: origin.z + z,
        };
        const partialDataOffset = this.getTexelOffset(textureDataLayout, format, point);
        const fullDataOffset = this.getTexelOffset(
          { offset: 0, bytesPerRow, rowsPerImage },
          format,
          point
        );
        this.device.defaultQueue.writeBuffer(
          fullData,
          fullDataOffset,
          partialData,
          partialDataOffset,
          (size.width / kTextureFormatInfo[format].blockWidth!) *
            kTextureFormatInfo[format].bytesPerBlock!
        );
      }
    }
  }

  // We check that the copied region of the texture matches data.
  // Used primarily for testing CopyT2B.
  partialCheck(
    textureCopyView: GPUTextureCopyView,
    textureDataLayout: GPUTextureDataLayout,
    size: GPUExtent3D,
    data: Uint8Array
  ): void {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(textureCopyView, { buffer, ...textureDataLayout }, size);
    this.device.defaultQueue.submit([encoder.finish()]);

    this.expectContents(buffer, data);
  }

  // We check that the whole texture matches data.
  // Used primarily for testing WriteTexture and CopyB2T.
  fullCheck(
    textureCopyView: GPUTextureCopyView,
    fullTextureCopyLayout: TextureCopyLayout,
    fullData: GPUBuffer
  ): void {
    const { texture, mipLevel } = textureCopyView;
    const { rowsPerImage, bytesPerRow, mipSize, byteLength } = fullTextureCopyLayout;

    const buffer = this.device.createBuffer({
      size: fullTextureCopyLayout.byteLength,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture, mipLevel },
      { buffer, rowsPerImage, bytesPerRow },
      mipSize
    );
    this.device.defaultQueue.submit([encoder.finish()]);

    this.expectEqualBuffers(buffer, 0, fullData, 0, byteLength);
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

        this.fillDataWithZeros(textureDataLayout, textureDesc.format, size, data);

        this.partialCheck(textureCopyView, textureDataLayout, size, data);

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

        this.updateFullData(
          textureDesc.format,
          fullTextureCopyLayout,
          textureDataLayout,
          size,
          origin,
          data,
          fullData
        );

        this.fullCheck(textureCopyView, fullTextureCopyLayout, fullData);

        break;
      }
      default:
        unreachable();
    }
  }
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

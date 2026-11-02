/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

//================================================================================================
// ann_rt.js: a small inference runtime for float models exported by ann_export_js()
//
// Copy this file next to the generated model file (which imports it). It is a plain ES module
// with no dependencies, for browsers and Node. It ports the float path of ann_rt.h function by
// function, with the same layer description (ann_rt_layer), so a model behaves as it does in
// C: values are stored as float32 (typed arrays) but sums and activations use JavaScript's
// double math, so results agree with ann_rt.h to float32 rounding, not bit for bit.
//================================================================================================

// the ann_rt.h version whose layer description this runtime reads
export const VERSION = 7;

// layer kinds and activations: the values of ann_rt.h's enums
export const DENSE = 0, CONV2D = 1, MAXPOOL2D = 2, FLATTEN = 3, UPSAMPLE2D = 4,
	GLOBAL_AVGPOOL2D = 5, GLOBAL_MAXPOOL2D = 6, EMBEDDING = 7;
export const ACT_NONE = 0, ACT_RELU = 1, ACT_LEAKY_RELU = 2, ACT_SIGMOID = 3, ACT_TANH = 4,
	ACT_SOFTSIGN = 5, ACT_GELU = 6, ACT_SOFTMAX = 7;

//------------------------------
// base64 of little-endian float32 values (as ann_export_js writes them) to a Float32Array
//------------------------------
export function decodeFloats(base64) {
	const text = atob(base64);
	const bytes = new Uint8Array(text.length);
	for (let i = 0; i < text.length; i++)
		bytes[i] = text.charCodeAt(i);
	const view = new DataView(bytes.buffer);
	const values = new Float32Array(bytes.length / 4);
	for (let i = 0; i < values.length; i++)
		values[i] = view.getFloat32(4 * i, true);
	return values;
}

//------------------------------
// a conv or pool layer's window width, stride and padding along the width
//------------------------------
const sizeW = l => l.size_w > 0 ? l.size_w : l.size;
const strideW = l => l.size_w > 0 ? l.stride_w : l.stride;
const padW = l => l.size_w > 0 ? l.pad_w : l.pad;

//------------------------------
// a conv layer's input channels per group, and the first one filter f reads
//------------------------------
const groupChannels = l => l.groups > 1 ? l.in_c / l.groups : l.in_c;
const firstChannel = (l, f) => l.groups > 1 ? Math.floor(f / (l.out_c / l.groups)) * groupChannels(l) : 0;

//------------------------------
// the token id at step t, rounded as libann reads ids; -1 for one outside the vocabulary
//------------------------------
function tokenId(l, ids, t) {
	const v = ids[t];
	return v >= -0.5 && v < l.size - 0.5 ? Math.floor(v + 0.5) : -1;
}

//------------------------------
// a masked layer's mask (1 kept, 0 padding, a byte per output step): an
// embedding's from its ids, a 1-D conv's or pool's from the layer below's
//------------------------------
function makeMask(l, ids, inMask, mask) {
	const k = sizeW(l), s = strideW(l), p = padW(l);

	if (l.kind === EMBEDDING) {
		for (let j = 0; j < l.out_w; j++)
			mask[j] = tokenId(l, ids, j) !== l.pad ? 1 : 0;
		return;
	}
	for (let j = 0; j < l.out_w; j++) {
		let lo = j * s, hi = j * s + k - 1 - 2 * p, keep = 1;

		if (hi < lo)		// a window narrower than its padding: its centre input
			lo = hi = j * s - p + Math.floor((k - 1) / 2);
		if (lo < 0)
			lo = 0;
		if (hi >= l.in_w)
			hi = l.in_w - 1;
		for (let i = lo; i <= hi; i++)
			if (!inMask[i])
				keep = 0;
		mask[j] = keep;
	}
}

function activate(act, x) {
	switch (act) {
	case ACT_RELU:			return x > 0 ? x : 0;
	case ACT_LEAKY_RELU:	return x > 0 ? x : 0.01 * x;
	case ACT_SIGMOID:		return 1 / (1 + Math.exp(-x));
	case ACT_TANH:			return Math.tanh(x);
	case ACT_SOFTSIGN:		return x / (1 + Math.abs(x));
	case ACT_GELU:			return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
	default:				return x;
	}
}

function softmax(y, n) {
	let max = y[0], sum = 0;
	for (let i = 1; i < n; i++)
		if (y[i] > max)
			max = y[i];
	for (let i = 0; i < n; i++) {
		y[i] = Math.exp(y[i] - max);
		sum += y[i];
	}
	for (let i = 0; i < n; i++)
		y[i] /= sum;
}

//------------------------------
// the linear part of a dense or conv layer: z = W x + b
//------------------------------
function linear(l, x, z) {
	const w = l.w, b = l.b;

	if (l.kind === DENSE) {
		const nIn = l.in_c * l.in_h * l.in_w;
		for (let o = 0; o < l.out_c; o++) {
			let acc = b[o];
			for (let i = 0, wi = o * nIn; i < nIn; i++, wi++)
				acc += x[i] * w[wi];
			z[o] = acc;
		}
		return;
	}

	// convolution, one kernel tap at a time over whole output rows (each
	// output still sums its terms in ann_rt.h's order: bias, then channel,
	// kernel row, kernel column), so the inner loop has no bounds checks
	const k = l.size, s = l.stride, p = l.pad, cg = groupChannels(l);
	const kw = sizeW(l), sw = strideW(l), pw = padW(l);
	const inH = l.in_h, inW = l.in_w, area = inH * inW;
	const outC = l.out_c, outH = l.out_h, outW = l.out_w, outArea = outH * outW;
	// the output columns whose input column ow * sw - pw + j is inside the image, for each j
	const lo = new Int32Array(kw), hi = new Int32Array(kw);
	for (let j = 0; j < kw; j++) {
		lo[j] = Math.max(0, Math.ceil((pw - j) / sw));
		hi[j] = Math.min(outW, Math.floor((inW - 1 + pw - j) / sw) + 1);
	}
	for (let f = 0; f < outC; f++) {
		const zf = f * outArea, xf = firstChannel(l, f) * area;
		z.fill(b[f], zf, zf + outArea);
		for (let c = 0; c < cg; c++) {
			const wc = (f * cg + c) * k * kw, xc = xf + c * area;
			for (let oh = 0; oh < outH; oh++) {
				const zr = zf + oh * outW;
				for (let i = 0; i < k; i++) {
					const ih = oh * s - p + i;
					if (ih < 0 || ih >= inH)
						continue;
					const xr = xc + ih * inW, wr = wc + i * kw;
					for (let j = 0; j < kw; j++) {
						const wv = w[wr + j], off = xr - pw + j, end = hi[j];
						if (sw === 1)
							for (let ow = lo[j]; ow < end; ow++)
								z[zr + ow] += x[off + ow] * wv;
						else
							for (let ow = lo[j]; ow < end; ow++)
								z[zr + ow] += x[off + ow * sw] * wv;
					}
				}
			}
		}
	}
}

function maxpool(l, x, y) {
	const kw = sizeW(l), sw = strideW(l);
	for (let c = 0; c < l.out_c; c++)
		for (let oh = 0; oh < l.out_h; oh++)
			for (let ow = 0; ow < l.out_w; ow++) {
				const xc = (c * l.in_h + oh * l.stride) * l.in_w + ow * sw;
				let best = x[xc];
				for (let i = 0; i < l.size; i++)
					for (let j = 0; j < kw; j++)
						if (x[xc + i * l.in_w + j] > best)
							best = x[xc + i * l.in_w + j];
				y[(c * l.out_h + oh) * l.out_w + ow] = best;
			}
}

function upsample(l, x, y) {
	let o = 0;
	for (let row = 0; row < l.in_c * l.in_h; row++) {
		const src = row * l.in_w;
		for (let r = 0; r < l.size; r++)
			for (let iw = 0; iw < l.in_w; iw++)
				for (let k = 0; k < l.size; k++)
					y[o++] = x[src + iw];
	}
}

function globalAvgpool(l, x, y, mask) {
	const area = l.in_h * l.in_w;
	for (let c = 0; c < l.out_c; c++) {
		let sum = 0, n = 0;
		for (let i = 0; i < area; i++)
			if (!mask || mask[i]) {
				sum += x[c * area + i];
				n++;
			}
		y[c] = n ? sum / n : 0;
	}
}

function globalMaxpool(l, x, y, mask) {
	const area = l.in_h * l.in_w;
	for (let c = 0; c < l.out_c; c++) {
		let best = 0, any = false;
		for (let i = 0; i < area; i++)
			if ((!mask || mask[i]) && (!any || x[c * area + i] > best)) {
				best = x[c * area + i];
				any = true;
			}
		y[c] = best;
	}
}

function embedding(l, ids, y) {
	const steps = l.out_w;
	for (let t = 0; t < steps; t++) {
		const id = tokenId(l, ids, t);
		const row = id >= 0 && id !== l.pad ? id * l.out_c : -1;
		for (let d = 0; d < l.out_c; d++)
			y[d * steps + t] = row >= 0 ? l.w[row + d] : 0;
	}
}

function maskZero(l, mask, y) {
	for (let t = 0; t < l.out_w; t++)
		if (!mask[t])
			for (let c = 0; c < l.out_c; c++)
				y[c * l.out_w + t] = 0;
}

//------------------------------
// float inference for one sample: input holds model.input_size values (token
// ids for a model that starts with an embedding). Returns a new Float32Array of
// model.output_size values, with softmax applied if the model has it. The
// model keeps its workspace between calls.
//------------------------------
export function predict(model, input) {
	if (input.length !== model.input_size)
		throw new RangeError(`expected ${model.input_size} input values, got ${input.length}`);
	if (!model.workspace)
		model.workspace = [new Float32Array(model.max_size), new Float32Array(model.max_size),
			new Uint8Array(model.mask_steps), new Uint8Array(model.mask_steps)];
	const [bufA, bufB, maskA, maskB] = model.workspace;
	let src = input, inMask = null;

	for (let n = 0; n < model.layers.length; n++) {
		const l = model.layers[n];
		const dst = src === bufA ? bufB : bufA;

		if ((l.kind === EMBEDDING) !== (n === 0 && model.layers[0].kind === EMBEDDING))
			throw new Error(`layer ${n}: an embedding must be the first layer`);
		switch (l.kind) {
		case DENSE:
		case CONV2D: {
			linear(l, src, dst);
			if (l.act !== ACT_SOFTMAX && l.act !== ACT_NONE) {
				const count = l.out_c * l.out_h * l.out_w;
				for (let i = 0; i < count; i++)
					dst[i] = activate(l.act, dst[i]);
			}
			break;
		}
		case MAXPOOL2D:			maxpool(l, src, dst); break;
		case UPSAMPLE2D:		upsample(l, src, dst); break;
		case GLOBAL_AVGPOOL2D:	globalAvgpool(l, src, dst, inMask); break;
		case GLOBAL_MAXPOOL2D:	globalMaxpool(l, src, dst, inMask); break;
		case EMBEDDING:			embedding(l, input, dst); break;
		case FLATTEN:			inMask = null; continue;
		default:				throw new Error(`layer ${n}: unknown kind ${l.kind}`);
		}
		if (l.masked) {
			const mask = inMask === maskA ? maskB : maskA;
			makeMask(l, input, inMask, mask);
			maskZero(l, mask, dst);
			inMask = mask;
		}
		else
			inMask = null;
		src = dst;
	}

	const output = Float32Array.from(src.subarray ? src.subarray(0, model.output_size) : src.slice(0, model.output_size));
	if (model.softmax)
		softmax(output, model.output_size);
	return output;
}

//------------------------------
// the index of the largest value (the class, for a classifier)
//------------------------------
export function argmax(values) {
	let best = 0;
	for (let i = 1; i < values.length; i++)
		if (values[i] > values[best])
			best = i;
	return best;
}

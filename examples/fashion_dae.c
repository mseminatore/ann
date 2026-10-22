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

//------------------------------------------------------------------------------
// A denoising autoencoder for Fashion-MNIST.
//
// A convolutional autoencoder learns to output clean images from noisy ones:
//
//   input 1x28x28 -> conv 16@3x3 relu -> maxpool 2 -> conv 8@3x3 relu
//   -> maxpool 2 (8x7x7) -> upsample 2 -> conv 16@3x3 relu -> upsample 2
//   -> conv 1@3x3 sigmoid (1x28x28)
//
// Training corrupts every batch with ann_set_input_noise() while the targets
// stay clean. The test images are corrupted the same way, and the example
// reports how close the noisy and the denoised images are to the clean ones
// (mean squared error and PSNR), then writes a picture of ten test images:
// clean, noisy and denoised rows, as fashion_dae.pgm and fashion_dae.png.
//
// Usage: fashion_dae [-e epochs] [-b batch] [-n gaussian|masking|saltpepper]
//                    [-a amount] [-r rows] [-o name] [-g] [train.csv] [test.csv]
// -a is the noise's standard deviation, or the fraction of pixels changed
// (default 0.3); -r trains on the first rows only; -o sets the image file name.
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

#define SIDE 28			// image width and height
#define PIXELS (SIDE * SIDE)
#define ZOOM 2			// each pixel is drawn as ZOOM x ZOOM in the picture
#define GAP 4			// pixels between pictured images

static int batch_size = 32;
static int epoch_count = 5;
static Noise_type noise_type = NOISE_GAUSSIAN;
static real noise_amount = (real)0.3;
static int train_rows = 0;	// 0 = all
static const char *image_name = "fashion_dae";
static int use_gpu = 0;

//----------------------------------
// an option's value: attached (-e2) or the next argument (-e 2)
//----------------------------------
static const char *option_value(int n, char *args[], int *i)
{
	if (args[*i][2])
		return &args[*i][2];
	if (*i + 1 < n)
		return args[++*i];
	return NULL;
}

//----------------------------------
// get options from the command line
//----------------------------------
static int get_options(int n, char *args[])
{
	int i;
	for (i = 1; (i < n) && (args[i][0] == '-'); i++)
	{
		const char *value;
		if (args[i][1] == 'b' && (value = option_value(n, args, &i)))
			batch_size = atoi(value);
		else if (args[i][1] == 'e' && (value = option_value(n, args, &i)))
			epoch_count = atoi(value);
		else if (args[i][1] == 'a' && (value = option_value(n, args, &i)))
			noise_amount = (real)atof(value);
		else if (args[i][1] == 'r' && (value = option_value(n, args, &i)))
			train_rows = atoi(value);
		else if (args[i][1] == 'o' && (value = option_value(n, args, &i)))
			image_name = value;
		else if (args[i][1] == 'n' && (value = option_value(n, args, &i)))
		{
			if (strcmp(value, "masking") == 0)
				noise_type = NOISE_MASKING;
			else if (strcmp(value, "saltpepper") == 0)
				noise_type = NOISE_SALT_PEPPER;
			else
				noise_type = NOISE_GAUSSIAN;
		}
		else if (args[i][1] == 'g')
			use_gpu = 1;
	}
	return i;
}

//----------------------------------
// load a label-first CSV into images scaled to [0, 1] and their classes
//----------------------------------
static int load_images(const char *filename, PTensor *x, PTensor *labels)
{
	real *data = NULL;
	int rows, stride;

	if (ERR_OK != ann_load_csv(filename, CSV_HAS_HEADER, &data, &rows, &stride))
	{
		printf("Error: unable to open data file - %s\n", filename);
		return ERR_FAIL;
	}

	*labels = tensor_create_from_array(rows, stride, data);
	free(data);

	*x = tensor_slice_cols(*labels, 1);	// pixels; labels keeps the class column
	tensor_mul_scalar(*x, (real)(1.0 / 255.0));
	return ERR_OK;
}

//----------------------------------
// mean squared difference over count values
//----------------------------------
static double mse(const real *a, const real *b, size_t count)
{
	double sum = 0.0;
	for (size_t i = 0; i < count; i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	return sum / (double)count;
}

//----------------------------------
// PSNR in dB for pixels in [0, 1]
//----------------------------------
static double psnr(double mse_value)
{
	return 10.0 * log10(1.0 / mse_value);
}

//==================================
// Image files: binary PGM, and PNG with uncompressed (stored) deflate blocks,
// so no compression library is needed
//==================================
static int write_pgm(const char *filename, const unsigned char *pixels, int width, int height)
{
	FILE *f = fopen(filename, "wb");
	if (!f)
		return ERR_IO;
	fprintf(f, "P5\n%d %d\n255\n", width, height);
	fwrite(pixels, 1, (size_t)width * height, f);
	return fclose(f) == 0 ? ERR_OK : ERR_IO;
}

static unsigned long crc32_update(unsigned long crc, const unsigned char *data, size_t count)
{
	crc = ~crc & 0xFFFFFFFFUL;
	for (size_t i = 0; i < count; i++)
	{
		crc ^= data[i];
		for (int k = 0; k < 8; k++)
			crc = (crc >> 1) ^ (0xEDB88320UL & (0UL - (crc & 1)));
	}
	return ~crc & 0xFFFFFFFFUL;
}

static void put_u32(unsigned char *p, unsigned long v)
{
	p[0] = (unsigned char)(v >> 24);
	p[1] = (unsigned char)(v >> 16);
	p[2] = (unsigned char)(v >> 8);
	p[3] = (unsigned char)v;
}

// a PNG chunk: length, type, data, CRC of type and data
static void write_chunk(FILE *f, const char *type, const unsigned char *data, size_t count)
{
	unsigned char head[8], tail[4];
	put_u32(head, (unsigned long)count);
	memcpy(head + 4, type, 4);
	fwrite(head, 1, 8, f);
	if (count)
		fwrite(data, 1, count, f);
	put_u32(tail, crc32_update(crc32_update(0, head + 4, 4), data, count));
	fwrite(tail, 1, 4, f);
}

static int write_png(const char *filename, const unsigned char *pixels, int width, int height)
{
	static const unsigned char signature[8] = { 137, 'P', 'N', 'G', '\r', '\n', 26, '\n' };
	size_t raw_size = (size_t)(width + 1) * height;		// each row starts with filter type 0
	size_t blocks = (raw_size + 65534) / 65535;
	unsigned char *raw = malloc(raw_size);
	unsigned char *zlib = malloc(2 + raw_size + 5 * blocks + 4);
	unsigned char ihdr[13];
	FILE *f = fopen(filename, "wb");

	if (!raw || !zlib || !f)
	{
		free(raw);
		free(zlib);
		if (f)
			fclose(f);
		return ERR_IO;
	}

	for (int y = 0; y < height; y++)
	{
		raw[(size_t)y * (width + 1)] = 0;
		memcpy(&raw[(size_t)y * (width + 1) + 1], &pixels[(size_t)y * width], width);
	}

	// zlib stream: header, stored blocks of at most 65535 bytes, Adler-32
	size_t n = 0;
	unsigned long a = 1, b = 0;
	zlib[n++] = 0x78;
	zlib[n++] = 0x01;
	for (size_t pos = 0; pos < raw_size; pos += 65535)
	{
		size_t len = raw_size - pos < 65535 ? raw_size - pos : 65535;
		zlib[n++] = pos + len == raw_size;	// final block flag, block type 0 (stored)
		zlib[n++] = (unsigned char)len;
		zlib[n++] = (unsigned char)(len >> 8);
		zlib[n++] = (unsigned char)~len;
		zlib[n++] = (unsigned char)(~len >> 8);
		memcpy(&zlib[n], &raw[pos], len);
		n += len;
	}
	for (size_t i = 0; i < raw_size; i++)
	{
		a = (a + raw[i]) % 65521;
		b = (b + a) % 65521;
	}
	put_u32(&zlib[n], (b << 16) | a);
	n += 4;

	put_u32(ihdr, (unsigned long)width);
	put_u32(ihdr + 4, (unsigned long)height);
	ihdr[8] = 8;	// bits per pixel
	ihdr[9] = 0;	// grayscale
	ihdr[10] = ihdr[11] = ihdr[12] = 0;	// deflate, adaptive filtering, no interlace

	fwrite(signature, 1, sizeof(signature), f);
	write_chunk(f, "IHDR", ihdr, sizeof(ihdr));
	write_chunk(f, "IDAT", zlib, n);
	write_chunk(f, "IEND", NULL, 0);

	free(raw);
	free(zlib);
	return fclose(f) == 0 ? ERR_OK : ERR_IO;
}

//----------------------------------
// draw image (PIXELS values in [0, 1]) at grid cell (col, row)
//----------------------------------
static void draw(unsigned char *picture, int width, int col, int row, const real *image)
{
	int x0 = GAP + col * (SIDE * ZOOM + GAP), y0 = GAP + row * (SIDE * ZOOM + GAP);

	for (int y = 0; y < SIDE * ZOOM; y++)
		for (int x = 0; x < SIDE * ZOOM; x++)
		{
			real v = image[(y / ZOOM) * SIDE + x / ZOOM];
			v = v < 0 ? 0 : (v > 1 ? 1 : v);
			picture[(size_t)(y0 + y) * width + x0 + x] = (unsigned char)(v * 255.0 + 0.5);
		}
}

static PNetwork make_network(void)
{
	PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_BINARY_CROSS_ENTROPY);

	ann_add_input(pnet, 1, SIDE, SIDE);
	ann_add_conv2d(pnet, 16, 3, 1, 1, ACTIVATION_RELU);		// 16x28x28
	ann_add_maxpool2d(pnet, 2, 2);							// 16x14x14
	ann_add_conv2d(pnet, 8, 3, 1, 1, ACTIVATION_RELU);		// 8x14x14
	ann_add_maxpool2d(pnet, 2, 2);							// 8x7x7
	ann_add_upsample2d(pnet, 2);							// 8x14x14
	ann_add_conv2d(pnet, 16, 3, 1, 1, ACTIVATION_RELU);		// 16x14x14
	ann_add_upsample2d(pnet, 2);							// 16x28x28
	ann_add_conv2d(pnet, 1, 3, 1, 1, ACTIVATION_SIGMOID);	// 1x28x28

	ann_set_epoch_limit(pnet, epoch_count);
	ann_set_convergence(pnet, (real)0.0);	// always train every epoch
	ann_set_batch_size(pnet, batch_size);
	ann_set_learning_rate(pnet, (real)0.002);

	// noisy inputs, clamped to the pixel range; the targets stay clean
	if (ann_set_input_noise(pnet, noise_type, noise_amount, (real)0.0, (real)1.0) != ERR_OK)
	{
		printf("Error: invalid noise amount %g\n", (double)noise_amount);
		ann_free_network(pnet);
		return NULL;
	}
	return pnet;
}

int main(int argc, char *argv[])
{
	int first_arg = get_options(argc, argv);
	const char *train_file = first_arg < argc ? argv[first_arg] : "fashion-mnist_train.csv";
	const char *test_file = first_arg + 1 < argc ? argv[first_arg + 1] : "fashion-mnist_test.csv";
	static const char *noise_names[] = { "none", "Gaussian", "masking", "salt-and-pepper" };

#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
#endif

	PTensor x_train, labels_train, x_test, labels_test;
	if (load_images(train_file, &x_train, &labels_train) != ERR_OK ||
		load_images(test_file, &x_test, &labels_test) != ERR_OK)
		return ERR_FAIL;

	PNetwork pnet = make_network();
	if (!pnet)
		return ERR_FAIL;

#ifdef USE_GPU
	if (use_gpu && (!ann_gpu_init() || ERR_OK != ann_gpu_upload_network(pnet)))
		puts("GPU unavailable for this network, training on CPU");
#else
	if (use_gpu)
		puts("GPU support not compiled in, using CPU");
#endif

	// -r: train on the first rows only
	if (train_rows > 0 && train_rows < x_train->rows)
	{
		PTensor first = tensor_create_from_array(train_rows, x_train->cols, x_train->values);
		tensor_free(x_train);
		x_train = first;
	}
	printf("Denoising %s noise (amount %g), training on %d images\n",
	       noise_names[noise_type], (double)noise_amount, x_train->rows);

	// the targets are the clean inputs
	ann_train_network(pnet, x_train, x_train, x_train->rows);

	// corrupt the test images like the training inputs, then denoise them
	size_t count = (size_t)x_test->rows * PIXELS;
	real *noisy = malloc(count * sizeof(real));
	real *denoised = malloc(count * sizeof(real));
	if (!noisy || !denoised)
		return ERR_ALLOC;
	memcpy(noisy, x_test->values, count * sizeof(real));
	srand(1);
	ann_apply_input_noise(pnet, noisy, (int)count);
	ann_predict_batch(pnet, noisy, denoised, x_test->rows);

	double noisy_mse = mse(noisy, x_test->values, count), denoised_mse = mse(denoised, x_test->values, count);
	printf("\nTest images (%d), compared with the clean ones:\n", x_test->rows);
	printf("  noisy:    MSE %.5f, PSNR %.2f dB\n", noisy_mse, psnr(noisy_mse));
	printf("  denoised: MSE %.5f, PSNR %.2f dB\n", denoised_mse, psnr(denoised_mse));

	// one test image of each class: clean, noisy and denoised rows
	int width = 10 * (SIDE * ZOOM + GAP) + GAP, height = 3 * (SIDE * ZOOM + GAP) + GAP;
	unsigned char *picture = malloc((size_t)width * height);
	if (!picture)
		return ERR_ALLOC;
	memset(picture, 128, (size_t)width * height);
	for (int cls = 0; cls < 10; cls++)
		for (int r = 0; r < x_test->rows; r++)
			if ((int)labels_test->values[(size_t)r * labels_test->cols] == cls)
			{
				draw(picture, width, cls, 0, &x_test->values[(size_t)r * PIXELS]);
				draw(picture, width, cls, 1, &noisy[(size_t)r * PIXELS]);
				draw(picture, width, cls, 2, &denoised[(size_t)r * PIXELS]);
				break;
			}

	char pgm[256], png[256];
	snprintf(pgm, sizeof(pgm), "%s.pgm", image_name);
	snprintf(png, sizeof(png), "%s.png", image_name);
	if (write_pgm(pgm, picture, width, height) == ERR_OK && write_png(png, picture, width, height) == ERR_OK)
		printf("\nWrote %s and %s (rows: clean, noisy, denoised)\n", pgm, png);
	else
		printf("\nError: unable to write %s or %s\n", pgm, png);

	free(picture);
	free(noisy);
	free(denoised);
	ann_free_network(pnet);
	tensor_free(x_train);
	tensor_free(labels_train);
	tensor_free(x_test);
	tensor_free(labels_test);
	return 0;
}

//-----------------------------------------------------------
// ann_load_csv: numbers separated by commas, spaces or tabs, with \r\n line
// endings, blank lines, quoted numbers and lines of any length; text, ragged
// rows and missing headers are rejected with a message naming the file line.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"
#include "testy/test.h"

static char g_msg[256];
static int g_code;

static void record_error(int code, const char *message, const char *function)
{
    (void)function;
    g_code = code;
    snprintf(g_msg, sizeof(g_msg), "%s", message);
}

// write text to path exactly (binary, so \n stays \n on Windows)
static void write_file(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

// load path; returns the error code, with *rows, *cols and the values in *data
static int load(const char *path, int header, real **data, int *rows, int *cols)
{
    g_msg[0] = '\0';
    g_code = ERR_OK;
    *data = NULL;
    return ann_load_csv(path, header, data, rows, cols);
}

static int values_are(const real *data, const real *expected, int n)
{
    for (int i = 0; i < n; i++)
        if (data[i] != expected[i])
            return 0;
    return 1;
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    const char *path = "test_csv.csv";
    real *data;
    int rows, cols;

    MODULE("CSV Loading");
    ann_set_error_log_callback(record_error);

    SUITE("Numbers");
    {
        write_file(path, "a,b,c\n1,2,3\n4.5,-6,7e-1\n");
        TESTEX("A file with a header loads", load(path, CSV_HAS_HEADER, &data, &rows, &cols) == ERR_OK &&
               rows == 2 && cols == 3);
        real expected[] = { 1, 2, 3, 4.5f, -6, 0.7f };
        TESTEX("with its values row by row", data && values_are(data, expected, 6));
        ann_free_csv(data);

        write_file(path, "1, 2\t3\r\n\r\n4 ,5 ,6\r\n\n7,8,9");
        TESTEX("Spaces, tabs, \\r\\n, blank lines and no final line break are fine",
               load(path, CSV_NO_HEADER, &data, &rows, &cols) == ERR_OK && rows == 3 && cols == 3 &&
               data[3] == 4 && data[8] == 9);
        ann_free_csv(data);

        write_file(path, "\"1\",\"2.5\"\n\"3\",4\n");
        TESTEX("Quoted numbers load", load(path, CSV_NO_HEADER, &data, &rows, &cols) == ERR_OK && rows == 2 &&
               data[1] == 2.5f && data[3] == 4);
        ann_free_csv(data);

        // 3000 values: longer than the 8192-character buffer lines used to be cut at
        FILE *f = fopen(path, "wb");
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3000; c++)
                fprintf(f, "%s%d.25", c ? "," : "", c);
            fputc('\n', f);
        }
        fclose(f);
        TESTEX("Lines of any length load", load(path, CSV_NO_HEADER, &data, &rows, &cols) == ERR_OK && rows == 2 &&
               cols == 3000 && data[5999] == 2999.25f);
        ann_free_csv(data);

        write_file(path, "a,b\n");
        TESTEX("A header with no rows gives none", load(path, CSV_HAS_HEADER, &data, &rows, &cols) == ERR_OK &&
               rows == 0 && cols == 0);
        ann_free_csv(data);
        ann_free_csv(NULL);
    }

    SUITE("Errors");
    {
        write_file(path, "review,sentiment\n1,positive\n\"One of the other reviewers has mentioned that, after\",positive\n");
        TESTEX("A text field is rejected", load(path, CSV_HAS_HEADER, &data, &rows, &cols) == ERR_INVALID &&
               data == NULL && rows == 0 && cols == 0);
        TESTEX("naming its file line (the header counts) and value",
               g_code == ERR_INVALID && strstr(g_msg, "test_csv.csv line 2, value 2: \"positive\" is not a number"));

        write_file(path, "1,2\n\"One of the other reviewers has mentioned that, after\",3\n");
        load(path, CSV_NO_HEADER, &data, &rows, &cols);
        TESTEX("Quoted text shows up to 40 characters",
               strstr(g_msg, "line 2, value 1: \"One of the other reviewers has mentioned...") != NULL);

        write_file(path, "1,2abc\n");
        TESTEX("and so is a number followed by text", load(path, CSV_NO_HEADER, &data, &rows, &cols) == ERR_INVALID &&
               strstr(g_msg, "\"2abc\"") != NULL);

        write_file(path, "h\n1,2,3\n\n4,5\n");
        TESTEX("A line with another number of values is rejected",
               load(path, CSV_HAS_HEADER, &data, &rows, &cols) == ERR_INVALID &&
               strstr(g_msg, "line 4 has 2 values, but line 2 has 3") != NULL);

        write_file(path, "label,pixel1\n");
        TESTEX("A header read as data names it", load(path, CSV_NO_HEADER, &data, &rows, &cols) == ERR_INVALID &&
               strstr(g_msg, "line 1, value 1: \"label\"") != NULL);

        write_file(path, "");
        TESTEX("An empty file has no header", load(path, CSV_HAS_HEADER, &data, &rows, &cols) == ERR_FAIL &&
               strstr(g_msg, "no header") != NULL);
        TESTEX("but loads as no rows without one", load(path, CSV_NO_HEADER, &data, &rows, &cols) == ERR_OK &&
               rows == 0);
        ann_free_csv(data);

        TESTEX("A missing file is ERR_IO", load("no_such_file.csv", CSV_NO_HEADER, &data, &rows, &cols) == ERR_IO);
    }

    ann_clear_error_log_callback();
    remove(path);
}

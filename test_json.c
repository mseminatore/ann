/**
 * @file test_json.c
 * @brief Tests for JSON parser
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "testy/test.h"
#include "json.h"

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    
    MODULE("JSON Parser Tests");
    
    // ========================================================================
    // BASIC VALUE TESTS
    // ========================================================================
    SUITE("Basic Values");
    COMMENT("Testing parsing of primitive JSON values...");
    
    JsonValue val;
    int result;
    
    // Null
    result = json_parse("null", &val);
    TESTEX("Parse null succeeds", (result == 0));
    TESTEX("Null type is JSON_NULL", (val.type == JSON_NULL));
    TESTEX("json_is_null returns 1", (json_is_null(&val) == 1));
    json_free(&val);
    
    // Boolean true
    result = json_parse("true", &val);
    TESTEX("Parse true succeeds", (result == 0));
    TESTEX("True type is JSON_BOOL", (val.type == JSON_BOOL));
    int b;
    TESTEX("json_bool returns 0", (json_bool(&val, &b) == 0));
    TESTEX("Boolean value is 1", (b == 1));
    json_free(&val);
    
    // Boolean false
    result = json_parse("false", &val);
    TESTEX("Parse false succeeds", (result == 0));
    TESTEX("json_bool returns 0", (json_bool(&val, &b) == 0));
    TESTEX("Boolean value is 0", (b == 0));
    json_free(&val);
    
    // Integer
    result = json_parse("42", &val);
    TESTEX("Parse integer succeeds", (result == 0));
    TESTEX("Integer type is JSON_NUMBER", (val.type == JSON_NUMBER));
    int i;
    TESTEX("json_int returns 0", (json_int(&val, &i) == 0));
    TESTEX("Integer value is 42", (i == 42));
    json_free(&val);
    
    // Negative number
    result = json_parse("-3.14159", &val);
    TESTEX("Parse negative float succeeds", (result == 0));
    double d;
    TESTEX("json_number returns 0", (json_number(&val, &d) == 0));
    TESTEX("Float value is approximately -3.14159", (d < -3.14 && d > -3.15));
    json_free(&val);
    
    // String
    result = json_parse("\"hello world\"", &val);
    TESTEX("Parse string succeeds", (result == 0));
    TESTEX("String type is JSON_STRING", (val.type == JSON_STRING));
    TESTEX("json_string returns non-NULL", (json_string(&val) != NULL));
    TESTEX("String value is 'hello world'", (strcmp(json_string(&val), "hello world") == 0));
    json_free(&val);
    
    // String with escapes
    result = json_parse("\"line1\\nline2\\ttab\"", &val);
    TESTEX("Parse string with escapes succeeds", (result == 0));
    TESTEX("Escaped string has newline", (strchr(json_string(&val), '\n') != NULL));
    TESTEX("Escaped string has tab", (strchr(json_string(&val), '\t') != NULL));
    json_free(&val);
    
    // ========================================================================
    // ARRAY TESTS
    // ========================================================================
    SUITE("Arrays");
    COMMENT("Testing array parsing...");
    
    // Empty array
    result = json_parse("[]", &val);
    TESTEX("Parse empty array succeeds", (result == 0));
    TESTEX("Empty array type is JSON_ARRAY", (val.type == JSON_ARRAY));
    TESTEX("Empty array length is 0", (json_array_length(&val) == 0));
    json_free(&val);
    
    // Simple array
    result = json_parse("[1, 2, 3]", &val);
    TESTEX("Parse [1,2,3] succeeds", (result == 0));
    TESTEX("Array length is 3", (json_array_length(&val) == 3));
    TESTEX("First element is 1", (json_at(&val, 0) != NULL));
    json_int(json_at(&val, 0), &i);
    TESTEX("First element value is 1", (i == 1));
    json_int(json_at(&val, 2), &i);
    TESTEX("Third element value is 3", (i == 3));
    json_free(&val);
    
    // Mixed array
    result = json_parse("[1, \"two\", true, null]", &val);
    TESTEX("Parse mixed array succeeds", (result == 0));
    TESTEX("Mixed array length is 4", (json_array_length(&val) == 4));
    TESTEX("Second element is string 'two'", 
           (strcmp(json_string(json_at(&val, 1)), "two") == 0));
    TESTEX("Fourth element is null", (json_is_null(json_at(&val, 3)) == 1));
    json_free(&val);
    
    // ========================================================================
    // OBJECT TESTS
    // ========================================================================
    SUITE("Objects");
    COMMENT("Testing object parsing...");
    
    // Empty object
    result = json_parse("{}", &val);
    TESTEX("Parse empty object succeeds", (result == 0));
    TESTEX("Empty object type is JSON_OBJECT", (val.type == JSON_OBJECT));
    json_free(&val);
    
    // Simple object
    result = json_parse("{\"name\": \"test\", \"value\": 123}", &val);
    TESTEX("Parse simple object succeeds", (result == 0));
    TESTEX("json_get('name') returns value", (json_get(&val, "name") != NULL));
    TESTEX("name is 'test'", (strcmp(json_string(json_get(&val, "name")), "test") == 0));
    json_int(json_get(&val, "value"), &i);
    TESTEX("value is 123", (i == 123));
    TESTEX("json_get('missing') returns NULL", (json_get(&val, "missing") == NULL));
    json_free(&val);
    
    // Nested object
    result = json_parse("{\"outer\": {\"inner\": 42}}", &val);
    TESTEX("Parse nested object succeeds", (result == 0));
    JsonValue *outer = json_get(&val, "outer");
    TESTEX("outer is an object", (outer != NULL && outer->type == JSON_OBJECT));
    JsonValue *inner = json_get(outer, "inner");
    json_int(inner, &i);
    TESTEX("inner value is 42", (i == 42));
    json_free(&val);
    
    // ========================================================================
    // COMPLEX STRUCTURE (ONNX-like)
    // ========================================================================
    SUITE("Complex Structure");
    COMMENT("Testing ONNX-like JSON structure...");
    
    const char *onnx_like = 
        "{\n"
        "  \"ir_version\": 8,\n"
        "  \"producer_name\": \"ann-library\",\n"
        "  \"graph\": {\n"
        "    \"initializer\": [\n"
        "      {\"name\": \"weight_0\", \"dims\": [4, 8], \"float_data\": [0.1, 0.2, 0.3]}\n"
        "    ],\n"
        "    \"node\": [\n"
        "      {\"op_type\": \"MatMul\", \"input\": [\"a\", \"b\"], \"output\": [\"c\"]}\n"
        "    ]\n"
        "  }\n"
        "}";
    
    result = json_parse(onnx_like, &val);
    TESTEX("Parse ONNX-like JSON succeeds", (result == 0));
    
    json_int(json_get(&val, "ir_version"), &i);
    TESTEX("ir_version is 8", (i == 8));
    
    TESTEX("producer_name is 'ann-library'", 
           (strcmp(json_string(json_get(&val, "producer_name")), "ann-library") == 0));
    
    JsonValue *graph = json_get(&val, "graph");
    TESTEX("graph exists", (graph != NULL));
    
    JsonValue *initializer = json_get(graph, "initializer");
    TESTEX("initializer is array", (initializer != NULL && initializer->type == JSON_ARRAY));
    TESTEX("initializer has 1 element", (json_array_length(initializer) == 1));
    
    JsonValue *weight = json_at(initializer, 0);
    TESTEX("weight name is 'weight_0'", 
           (strcmp(json_string(json_get(weight, "name")), "weight_0") == 0));
    
    JsonValue *dims = json_get(weight, "dims");
    TESTEX("dims has 2 elements", (json_array_length(dims) == 2));
    json_int(json_at(dims, 0), &i);
    TESTEX("First dim is 4", (i == 4));
    
    JsonValue *float_data = json_get(weight, "float_data");
    TESTEX("float_data has 3 elements", (json_array_length(float_data) == 3));
    json_number(json_at(float_data, 0), &d);
    TESTEX("First float is 0.1", (d > 0.09 && d < 0.11));
    
    JsonValue *node = json_get(graph, "node");
    JsonValue *matmul = json_at(node, 0);
    TESTEX("First node op_type is 'MatMul'", 
           (strcmp(json_string(json_get(matmul, "op_type")), "MatMul") == 0));
    
    json_free(&val);
    
    // ========================================================================
    // ERROR CASES
    // ========================================================================
    SUITE("Error Handling");
    COMMENT("Testing parse error handling...");
    
    result = json_parse("", &val);
    TESTEX("Empty string returns error", (result != 0));
    
    result = json_parse("{invalid}", &val);
    TESTEX("Invalid JSON returns error", (result != 0));
    
    result = json_parse("[1, 2,]", &val);
    TESTEX("Trailing comma returns error", (result != 0));
    
    result = json_parse("{\"key\"}", &val);
    TESTEX("Missing value returns error", (result != 0));
    
    result = json_parse(NULL, &val);
    TESTEX("NULL input returns error", (result != 0));
    
    TESTEX("JSON parser tests completed", 1);
}

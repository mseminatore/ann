/**
 * @file json.h
 * @brief Minimal JSON parser for ONNX import
 * 
 * A lightweight JSON parser sufficient for reading ONNX JSON files.
 * Supports: objects, arrays, strings, numbers, booleans, null.
 */

#ifndef JSON_H
#define JSON_H

#include <stddef.h>

//------------------------------
// JSON value types
//------------------------------
typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

//------------------------------
// JSON value structure
//------------------------------
typedef struct JsonValue JsonValue;
typedef struct JsonPair JsonPair;

struct JsonValue {
    JsonType type;
    union {
        int bool_val;           // JSON_BOOL
        double num_val;         // JSON_NUMBER
        char *str_val;          // JSON_STRING (owned, must free)
        struct {                // JSON_ARRAY
            JsonValue *items;
            size_t count;
        } array;
        struct {                // JSON_OBJECT
            JsonPair *pairs;
            size_t count;
        } object;
    } u;
};

struct JsonPair {
    char *key;                  // owned, must free
    JsonValue value;
};

//------------------------------
// Parser functions
//------------------------------

/**
 * Parse JSON from a null-terminated string.
 * 
 * @param json JSON text to parse
 * @param out Pointer to receive parsed value
 * @return 0 on success, -1 on parse error
 */
int json_parse(const char *json, JsonValue *out);

/**
 * Parse JSON from a file.
 * 
 * @param filename Path to JSON file
 * @param out Pointer to receive parsed value
 * @return 0 on success, -1 on error
 */
int json_parse_file(const char *filename, JsonValue *out);

/**
 * Free a JSON value and all nested values.
 * 
 * @param val JSON value to free
 */
void json_free(JsonValue *val);

//------------------------------
// Accessor functions
//------------------------------

/**
 * Get object member by key.
 * 
 * @param obj JSON object value
 * @param key Member key to find
 * @return Pointer to member value, or NULL if not found
 */
JsonValue *json_get(const JsonValue *obj, const char *key);

/**
 * Get array element by index.
 * 
 * @param arr JSON array value
 * @param index Element index
 * @return Pointer to element, or NULL if out of bounds
 */
JsonValue *json_at(const JsonValue *arr, size_t index);

/**
 * Get string value.
 * 
 * @param val JSON value (must be JSON_STRING)
 * @return String pointer, or NULL if not a string
 */
const char *json_string(const JsonValue *val);

/**
 * Get number value.
 * 
 * @param val JSON value (must be JSON_NUMBER)
 * @param out Pointer to receive number
 * @return 0 on success, -1 if not a number
 */
int json_number(const JsonValue *val, double *out);

/**
 * Get integer value.
 * 
 * @param val JSON value (must be JSON_NUMBER)
 * @param out Pointer to receive integer
 * @return 0 on success, -1 if not a number
 */
int json_int(const JsonValue *val, int *out);

/**
 * Get boolean value.
 * 
 * @param val JSON value (must be JSON_BOOL)
 * @param out Pointer to receive boolean (0 or 1)
 * @return 0 on success, -1 if not a boolean
 */
int json_bool(const JsonValue *val, int *out);

/**
 * Get array length.
 * 
 * @param val JSON value (must be JSON_ARRAY)
 * @return Array length, or 0 if not an array
 */
size_t json_array_length(const JsonValue *val);

/**
 * Check if value is null.
 * 
 * @param val JSON value
 * @return 1 if null, 0 otherwise
 */
int json_is_null(const JsonValue *val);

#endif /* JSON_H */

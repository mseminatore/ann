/**
 * @file json.c
 * @brief Minimal JSON parser for ONNX import
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "json.h"

//------------------------------
// Parser state
//------------------------------
typedef struct {
    const char *text;
    size_t pos;
    size_t len;
} Parser;

//------------------------------
// Forward declarations
//------------------------------
static int parse_value(Parser *p, JsonValue *out);
static void skip_whitespace(Parser *p);

//------------------------------
// Utility functions
//------------------------------
static char peek(Parser *p)
{
    if (p->pos >= p->len) return '\0';
    return p->text[p->pos];
}

static char advance(Parser *p)
{
    if (p->pos >= p->len) return '\0';
    return p->text[p->pos++];
}

static int match(Parser *p, const char *str)
{
    size_t len = strlen(str);
    if (p->pos + len > p->len) return 0;
    if (strncmp(&p->text[p->pos], str, len) == 0)
    {
        p->pos += len;
        return 1;
    }
    return 0;
}

static void skip_whitespace(Parser *p)
{
    while (p->pos < p->len)
    {
        char c = p->text[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            p->pos++;
        else
            break;
    }
}

//------------------------------
// Parse string
//------------------------------
static int parse_string(Parser *p, char **out)
{
    if (advance(p) != '"') return -1;
    
    // Find end of string and calculate length
    size_t start = p->pos;
    size_t len = 0;
    
    while (p->pos < p->len)
    {
        char c = p->text[p->pos];
        if (c == '"') break;
        if (c == '\\')
        {
            p->pos++;
            if (p->pos >= p->len) return -1;
        }
        p->pos++;
        len++;
    }
    
    if (peek(p) != '"') return -1;
    
    // Allocate and copy string, handling escapes
    char *str = (char *)malloc(len + 1);
    if (!str) return -1;
    
    size_t j = 0;
    for (size_t i = start; i < p->pos; i++)
    {
        char c = p->text[i];
        if (c == '\\' && i + 1 < p->pos)
        {
            i++;
            c = p->text[i];
            switch (c)
            {
            case 'n': str[j++] = '\n'; break;
            case 't': str[j++] = '\t'; break;
            case 'r': str[j++] = '\r'; break;
            case '"': str[j++] = '"'; break;
            case '\\': str[j++] = '\\'; break;
            case '/': str[j++] = '/'; break;
            default: str[j++] = c; break;
            }
        }
        else
        {
            str[j++] = c;
        }
    }
    str[j] = '\0';
    
    advance(p);  // consume closing quote
    *out = str;
    return 0;
}

//------------------------------
// Parse number
//------------------------------
static int parse_number(Parser *p, double *out)
{
    char *end;
    *out = strtod(&p->text[p->pos], &end);
    if (end == &p->text[p->pos]) return -1;
    p->pos = (size_t)(end - p->text);
    return 0;
}

//------------------------------
// Parse array
//------------------------------
static int parse_array(Parser *p, JsonValue *out)
{
    if (advance(p) != '[') return -1;
    
    out->type = JSON_ARRAY;
    out->u.array.items = NULL;
    out->u.array.count = 0;
    
    size_t capacity = 0;
    
    skip_whitespace(p);
    if (peek(p) == ']')
    {
        advance(p);
        return 0;
    }
    
    while (1)
    {
        // Grow array if needed
        if (out->u.array.count >= capacity)
        {
            capacity = capacity == 0 ? 8 : capacity * 2;
            JsonValue *new_items = (JsonValue *)realloc(out->u.array.items, 
                capacity * sizeof(JsonValue));
            if (!new_items)
            {
                json_free(out);
                return -1;
            }
            out->u.array.items = new_items;
        }
        
        skip_whitespace(p);
        if (parse_value(p, &out->u.array.items[out->u.array.count]) != 0)
        {
            json_free(out);
            return -1;
        }
        out->u.array.count++;
        
        skip_whitespace(p);
        if (peek(p) == ']')
        {
            advance(p);
            return 0;
        }
        if (advance(p) != ',')
        {
            json_free(out);
            return -1;
        }
    }
}

//------------------------------
// Parse object
//------------------------------
static int parse_object(Parser *p, JsonValue *out)
{
    if (advance(p) != '{') return -1;
    
    out->type = JSON_OBJECT;
    out->u.object.pairs = NULL;
    out->u.object.count = 0;
    
    size_t capacity = 0;
    
    skip_whitespace(p);
    if (peek(p) == '}')
    {
        advance(p);
        return 0;
    }
    
    while (1)
    {
        // Grow pairs array if needed
        if (out->u.object.count >= capacity)
        {
            capacity = capacity == 0 ? 8 : capacity * 2;
            JsonPair *new_pairs = (JsonPair *)realloc(out->u.object.pairs,
                capacity * sizeof(JsonPair));
            if (!new_pairs)
            {
                json_free(out);
                return -1;
            }
            out->u.object.pairs = new_pairs;
        }
        
        skip_whitespace(p);
        
        // Parse key
        char *key;
        if (parse_string(p, &key) != 0)
        {
            json_free(out);
            return -1;
        }
        out->u.object.pairs[out->u.object.count].key = key;
        
        skip_whitespace(p);
        if (advance(p) != ':')
        {
            free(key);
            json_free(out);
            return -1;
        }
        
        skip_whitespace(p);
        if (parse_value(p, &out->u.object.pairs[out->u.object.count].value) != 0)
        {
            free(key);
            json_free(out);
            return -1;
        }
        out->u.object.count++;
        
        skip_whitespace(p);
        if (peek(p) == '}')
        {
            advance(p);
            return 0;
        }
        if (advance(p) != ',')
        {
            json_free(out);
            return -1;
        }
    }
}

//------------------------------
// Parse any value
//------------------------------
static int parse_value(Parser *p, JsonValue *out)
{
    skip_whitespace(p);
    
    char c = peek(p);
    
    if (c == '"')
    {
        out->type = JSON_STRING;
        return parse_string(p, &out->u.str_val);
    }
    else if (c == '[')
    {
        return parse_array(p, out);
    }
    else if (c == '{')
    {
        return parse_object(p, out);
    }
    else if (c == 't' && match(p, "true"))
    {
        out->type = JSON_BOOL;
        out->u.bool_val = 1;
        return 0;
    }
    else if (c == 'f' && match(p, "false"))
    {
        out->type = JSON_BOOL;
        out->u.bool_val = 0;
        return 0;
    }
    else if (c == 'n' && match(p, "null"))
    {
        out->type = JSON_NULL;
        return 0;
    }
    else if (c == '-' || isdigit((unsigned char)c))
    {
        out->type = JSON_NUMBER;
        return parse_number(p, &out->u.num_val);
    }
    
    return -1;
}

//------------------------------
// Public API
//------------------------------

int json_parse(const char *json, JsonValue *out)
{
    if (!json || !out) return -1;
    
    Parser p;
    p.text = json;
    p.pos = 0;
    p.len = strlen(json);
    
    memset(out, 0, sizeof(JsonValue));
    
    if (parse_value(&p, out) != 0)
        return -1;
    
    skip_whitespace(&p);
    
    // Ensure we consumed all input
    if (p.pos != p.len)
    {
        json_free(out);
        return -1;
    }
    
    return 0;
}

int json_parse_file(const char *filename, JsonValue *out)
{
    if (!filename || !out) return -1;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size <= 0)
    {
        fclose(f);
        return -1;
    }
    
    // Read file
    char *buffer = (char *)malloc((size_t)size + 1);
    if (!buffer)
    {
        fclose(f);
        return -1;
    }
    
    size_t read = fread(buffer, 1, (size_t)size, f);
    fclose(f);
    
    buffer[read] = '\0';
    
    int result = json_parse(buffer, out);
    free(buffer);
    
    return result;
}

void json_free(JsonValue *val)
{
    if (!val) return;
    
    switch (val->type)
    {
    case JSON_STRING:
        free(val->u.str_val);
        val->u.str_val = NULL;
        break;
        
    case JSON_ARRAY:
        for (size_t i = 0; i < val->u.array.count; i++)
        {
            json_free(&val->u.array.items[i]);
        }
        free(val->u.array.items);
        val->u.array.items = NULL;
        val->u.array.count = 0;
        break;
        
    case JSON_OBJECT:
        for (size_t i = 0; i < val->u.object.count; i++)
        {
            free(val->u.object.pairs[i].key);
            json_free(&val->u.object.pairs[i].value);
        }
        free(val->u.object.pairs);
        val->u.object.pairs = NULL;
        val->u.object.count = 0;
        break;
        
    default:
        break;
    }
    
    val->type = JSON_NULL;
}

JsonValue *json_get(const JsonValue *obj, const char *key)
{
    if (!obj || obj->type != JSON_OBJECT || !key) return NULL;
    
    for (size_t i = 0; i < obj->u.object.count; i++)
    {
        if (strcmp(obj->u.object.pairs[i].key, key) == 0)
        {
            return &obj->u.object.pairs[i].value;
        }
    }
    return NULL;
}

JsonValue *json_at(const JsonValue *arr, size_t index)
{
    if (!arr || arr->type != JSON_ARRAY) return NULL;
    if (index >= arr->u.array.count) return NULL;
    return &arr->u.array.items[index];
}

const char *json_string(const JsonValue *val)
{
    if (!val || val->type != JSON_STRING) return NULL;
    return val->u.str_val;
}

int json_number(const JsonValue *val, double *out)
{
    if (!val || val->type != JSON_NUMBER || !out) return -1;
    *out = val->u.num_val;
    return 0;
}

int json_int(const JsonValue *val, int *out)
{
    if (!val || val->type != JSON_NUMBER || !out) return -1;
    *out = (int)val->u.num_val;
    return 0;
}

int json_bool(const JsonValue *val, int *out)
{
    if (!val || val->type != JSON_BOOL || !out) return -1;
    *out = val->u.bool_val;
    return 0;
}

size_t json_array_length(const JsonValue *val)
{
    if (!val || val->type != JSON_ARRAY) return 0;
    return val->u.array.count;
}

int json_is_null(const JsonValue *val)
{
    return val && val->type == JSON_NULL;
}

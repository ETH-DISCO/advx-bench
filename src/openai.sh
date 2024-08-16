#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -k|--key) key="$2"; shift ;;
        *) echo "unknown parameter passed, use --key flag: $1"; exit 1 ;;
    esac
    shift
done

echo -e "key=$key\n\n"

# see: https://platform.openai.com/docs/api-reference/images

curl https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $key" \
  -d '{
    "model": "dall-e-3",
    "prompt": "an image of a photorealistic city in a desert where all colors are inverted and a blue camel in the right with very small hoofs. psychedelic surrealism.",   
    "n": 1,           
    "size": "1024x1024"
  }'

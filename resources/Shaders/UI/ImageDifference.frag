#version 430
in vec2 uv;

out vec4 fragColor;

uniform sampler2D loadedImageTexture;
uniform sampler2D screenTexture;

void main()
{
    vec3 loadedImagePixelColor = texture(loadedImageTexture, uv).xyz;
    vec3 screenPixelColor = texture(screenTexture, uv).xyz;

    float differenceR = abs(loadedImagePixelColor.x - screenPixelColor.x) / ((loadedImagePixelColor.x + screenPixelColor.x) / 2);
    float differenceG = abs(loadedImagePixelColor.y - screenPixelColor.y) / ((loadedImagePixelColor.y + screenPixelColor.y) / 2);
    float differenceB = abs(loadedImagePixelColor.z - screenPixelColor.z) / ((loadedImagePixelColor.z + screenPixelColor.z) / 2);

    float averageDifference = (differenceR + differenceG + differenceB) / 3.0f;

    fragColor = vec4(averageDifference, 1.0f - averageDifference, 0.0f, 1.0f);
}
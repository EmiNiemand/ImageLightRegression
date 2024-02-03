#version 430
in vec2 uv;

out vec4 fragColor;

uniform sampler2D loadedImageTexture;
uniform sampler2D screenTexture;

void main()
{


    vec3 loadedImagePixelColor = texture(loadedImageTexture, uv).rgb;
    vec3 screenPixelColor = texture(screenTexture, uv).rgb;

    vec3 colorDiff = abs(loadedImagePixelColor.rgb - screenPixelColor.rgb);

    float averageDifference = (colorDiff.r + colorDiff.g + colorDiff.b) / 3.0f;
    float smoothDiff = smoothstep(0.0, 0.1, averageDifference);
    vec3 color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), smoothDiff);

    fragColor = vec4(color, 1.0f);
}
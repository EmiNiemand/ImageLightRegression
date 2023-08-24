#version 430

out vec4 fragColor;

in vec2 uv;

uniform sampler2D screenTexture;
uniform sampler2D selectedObjectTexture;
uniform ivec2 screenPosition;

// prewitt
mat3 sx = mat3(
1.0, 1.0, 1.0,
0.0, 0.0, 0.0,
-1.0, -1.0, -1.0
);

mat3 sy = mat3(
1.0, 0.0, -1.0,
1.0, 0.0, -1.0,
1.0, 0.0, -1.0
);

void main()
{

    vec3 diffuse = texture(screenTexture, uv.st).rgb;
    mat3 I;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            vec3 sam  = texelFetch(selectedObjectTexture, ivec2(gl_FragCoord) - screenPosition + ivec2(i-1, j-1), 0 ).rgb;
            I[i][j] = length(sam);
        }
    }

    float gx = dot(sx[0], I[0]) + dot(sx[1], I[1]) + dot(sx[2], I[2]);
    float gy = dot(sy[0], I[0]) + dot(sy[1], I[1]) + dot(sy[2], I[2]);

    float g = sqrt(pow(gx, 2.0) + pow(gy, 2.0));

    vec3 edgeColor = vec3(1.0, 0.0, 0.0);

    vec4 color = vec4(mix(diffuse, edgeColor, g), 1.0);

    fragColor = color;
}
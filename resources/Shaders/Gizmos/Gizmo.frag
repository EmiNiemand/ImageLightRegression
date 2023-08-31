#version 430
out vec4 fragColor;

void main() {
    if (gl_PrimitiveID == 0) {
        fragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    else if (gl_PrimitiveID == 1) {
        fragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    else if (gl_PrimitiveID == 2) {
        fragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    }
}

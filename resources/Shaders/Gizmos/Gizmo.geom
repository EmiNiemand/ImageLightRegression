#version 430

layout (points) in;
layout (line_strip, max_vertices = 100) out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform int mode = 0;


const float PI = 3.1415926;
const float detailLevel = 24;
float size = 1;

void DrawLine(int id, vec4 vertexPosition, vec4 direction) {
    gl_PrimitiveID = id;
    gl_Position = projection * view * vertexPosition;
    EmitVertex();
    gl_Position = projection * view * (vertexPosition + direction * 0.25f * size);
    EmitVertex();
    EndPrimitive();
}

void DrawCircle(int id, vec4 vertexPosition, vec3 direction1, vec3 direction2) {
    gl_PrimitiveID = id;
    for (int i = 1; i <= detailLevel + 1; i++) {
        float ang = PI * 2 / detailLevel * i;
        vec3 offset = (direction1 * cos(ang) + direction2 * -sin(ang));
        gl_Position = projection * view * (vertexPosition + vec4(offset, 0) * 0.25f * size);
        EmitVertex();
    }
    EndPrimitive();
}

void DrawClub(int id, vec4 vertexPosition, vec4 direction) {
    vec4 previousPosition = vertexPosition;
    float scale = 0.01f * size;
    float doubleScale = scale * 2;
    // box vectors
    vec4 forward = normalize(direction);
    vec4 side;
    if (forward.y <= 0.99f) {
        side = vec4(forward.z, forward.y, forward.x, 0);
    }
    else {
        side = vec4(forward.x, forward.z, forward.y, 0);
    }
    vec4 upward = vec4(normalize(cross(vec3(forward), vec3(side))), 0);

    gl_PrimitiveID = id;
    // start
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go forward
    previousPosition += direction * 0.25f * size;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go left
    previousPosition += side * scale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go forward
    previousPosition += forward * doubleScale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go right
    previousPosition -= side * scale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go up
    previousPosition += upward * scale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go back
    previousPosition -= forward * doubleScale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go down
    previousPosition -= upward * doubleScale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go forward
    previousPosition += forward * doubleScale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go up
    previousPosition += upward * scale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    //go right
    previousPosition -= side * scale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go back
    previousPosition -= forward * doubleScale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    // go left
    previousPosition += side * scale;
    gl_Position = projection * view * previousPosition;
    EmitVertex();
    EndPrimitive();
}

void main() {
    vec4 vertexPosition = model * gl_in[0].gl_Position;

    size = length(view[3].xyz) / 3;

    if (mode == 0) {
        DrawLine(0, vertexPosition, vec4(1, 0, 0, 0));
        DrawLine(1, vertexPosition, vec4(0, 1, 0, 0));
        DrawLine(2, vertexPosition, vec4(0, 0, 1, 0));
    }
    else if (mode == 1) {
        DrawCircle(0, vertexPosition, vec3(1, 0, 0), vec3(0, 0, 1));
        DrawCircle(1, vertexPosition, vec3(0, 1, 0), vec3(0, 0, 1));
        DrawCircle(2, vertexPosition, vec3(1, 0, 0), vec3(0, 1, 0));
    }
    else if (mode == 2) {
        DrawClub(0, vertexPosition, vec4(1, 0, 0, 0));
        DrawClub(1, vertexPosition, vec4(0, 1, 0, 0));
        DrawClub(2, vertexPosition, vec4(0, 0, 1, 0));
    }
}

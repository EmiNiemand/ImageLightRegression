#version 430

layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = aPos;
    mat4 newView = view;
    newView[3][0] = 0;
    newView[3][1] = 0;
    newView[3][2] = 0;
    vec4 pos = projection * newView * vec4(aPos, 1.0);
    gl_Position = vec4(pos.x, pos.y, pos.w, pos.w);
}

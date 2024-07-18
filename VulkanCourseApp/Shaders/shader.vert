#version 450    // Use GLSL 4.5

layout (location = 0) in vec3 pos;       // Input vertex position
layout (location = 1) in vec3 col;       // Input vertex color

layout (binding = 0) uniform UboViewProjection {
    mat4 projection;
    mat4 view;
} uboViewProjection;

layout(push_constant) uniform PushModel {
    mat4 model;
} pushModel;

layout (location = 0) out vec3 fragCol;  // Output color to fragment shader

void main() {
    gl_Position = uboViewProjection.projection * uboViewProjection.view * pushModel.model * vec4(pos, 1.0);        // Assign position to gl_Position
    fragCol = col;                       // Pass color to fragment shader
}
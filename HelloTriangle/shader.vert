#version 450
layout(location = 0) out vec4 color;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 inColor;

void main(){
color = vec4(inColor,1);
	gl_Position = vec4(position,1);
}
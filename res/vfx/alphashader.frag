#version 330

uniform vec3 viewRay;
uniform float k;
uniform float maxTime;
uniform vex3 color;

smooth in vec3 out_normal;
smooth in float out_alphaTime;
smooth in float	out_alphaCurvature;
flat in float alphaShape;
flat in float area;

out vec4 fragColor;

void main()
{
	float gamma=dot(out_normal,viewRay)/(sqrt(dot(out_normal,out_normal))*sqrt(dot(viewRay,viewRay)));

	alphaDensity=clamp(k/(area*gamma));
	alphaFade=clamp(1.0f-out_alphaTime/maxTime);

	fragColor= vec4(color,alphaDensity*alphaShape*alphaCurvature*alphaFade);
}
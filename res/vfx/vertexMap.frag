
 varying vec3 vs_out_vertPos;

void main()
{
	gl_Color=vec4(vs_out_vertPos,1);
}
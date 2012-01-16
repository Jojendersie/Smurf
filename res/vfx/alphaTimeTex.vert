 attribute float in_vertIndex;

 out float vertIndex;

 void main()
 {
	vertIndex=in_vertIndex;
	gl_Position=gl_Vertex();
 }
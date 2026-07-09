inputDir = getDirectory("Choose Input Directory");  //Choose input directory; with all single page tifs inside. No other folders or files in this directory.
outputDir = getDirectory("Choose an output directory");//Choose output directory; important: must be NOT inside the input directory

#@ Integer (label="Chunk size", style="slider", min=1000, max=5000, stepSize=2000) chunk_size
#@ String (visibility=MESSAGE, value="*Note: Chunk size must be an factor of total number of frames", required=false) msg
#@ Boolean (label="Perform resize") resize
#@ Integer (label="Size X") size_X
#@ Integer (label="Size Y") size_Y
//
subfiles=getFileList(inputDir);
sorted = Array.sort(subfiles);
//
//
tifs=newArray();
for (j=0; j<sorted.length;j++){
	if (endsWith(sorted[j],'.ome.tif')) 
		tifs=Array.concat(tifs,inputDir+sorted[j]);
	};
run("Image Sequence...","open="+tifs[0]+" sort");

chunk_TSeries(outputDir);

function chunk_TSeries(outputDir) {
	stackTitle=getTitle();
	Stack.getDimensions(width, height, channels, slices, frames);
	print(slices); //print number of slices in TSeries
	chunk_size=chunk_size;//chunk_size; //choose size of substack, slices must be divisible by chunk_size 
	num_chunks=slices/chunk_size;
	
	if (resize){
		print("Resizing");
		run("Size...", "width="+size_X+" height="+size_Y+" constrain average interpolation=Bilinear"); // if you want to spatially downsample
	}
	
	print("Splitting Stack" + stackTitle + " into " + num_chunks +" substacks");
	run("Stack Splitter", "number="+num_chunks);
	print("Done splitting");
	selectWindow(stackTitle); 
	close();
	File.makeDirectory(outputDir+stackTitle);
	savepath=outputDir+stackTitle;
	

	for (j=1;j<nImages+1;j++) { 
	        selectImage(j); 
	        title = getTitle; 
	        print("Saving substack "+j+" of "+ nImages);
	        saveAs("Tiff", savepath + "/" + title);
	        print("Saved substack " + title);
	        }
	run("Close All"); 
	print("Finished processing stack " + stackTitle);
}


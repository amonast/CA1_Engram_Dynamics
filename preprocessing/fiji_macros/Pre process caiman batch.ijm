// Macro to pre-process TSeries ome.tiffs into a series of small (500MB-2GB) multipage tiffs for Caiman 
// important: no white space in folder names --> throws error 
// NO META DATA FILES CAN BE IN THE TSeries Subfolders!! it will slow down fiji's loading of the Tseries.
#@ Integer (label="Chunk size", style="slider", min=1000, max=5000, stepSize=1000) chunk_size
#@ String (visibility=MESSAGE, value="*Note: Chunk size must be an factor of total number of frames", required=false) msg
#@ Boolean (label="Perform resize") resize
#@ Integer (label="Size X") size_X
#@ Integer (label="Size Y") size_Y

inputDir = getDirectory("Choose Input Directory");  //Choose input directory; with subfolders of all TSeries data. No other folders or files in this directory.
outputDir = getDirectory("Choose an output directory");//Choose output directory; important: must be NOT inside the input directory
listdir = getFileList(inputDir);
Array.print(listdir);

//setBatchMode(true);
for(i=0; i< listdir.length; i++) {
	print("Processing " + inputDir+listdir[i]);
	subfiles=getFileList(inputDir+listdir[i]);
	sorted = Array.sort(subfiles);
	print("making array");
	tifs=newArray();
	for (j=0; j<sorted.length;j++){
		if (endsWith(sorted[j],'.ome.tif')) 
			tifs=Array.concat(tifs,inputDir+listdir[i]+sorted[j]);
		}
	print(tifs[0]);
	run("Image Sequence...","open="+tifs[0]+" sort");
	chunk_TSeries(outputDir);
	}
//setBatchMode(true);
for(i=0; i< listdir.length; i++) {
	print("Processing " + inputDir+listdir[i]);
	subfiles=getFileList(inputDir+listdir[i]);
	sorted = Array.sort(subfiles);
	print("making array");
	tifs=newArray();
	for (j=0; j<sorted.length;j++){
		if (endsWith(sorted[j],'.ome.tif')) 
			tifs=Array.concat(tifs,inputDir+listdir[i]+sorted[j]);
		}
	print(tifs);
	run("Image Sequence...","open="+sorted[0]+" sort");
	chunk_TSeries(outputDir);
}
chunk_TSeries(outputDir);
print("All done!");

function chunk_TSeries(outputDir) {
	stackTitle=getTitle();
	Stack.getDimensions(width, height, channels, slices, frames);
	print(slices); //print number of slices in TSeries
	//chunk_size=2000; //choose size of substack, slices must be divisible by chunk_size 
	num_chunks=slices/chunk_size;
	if (resize)
		print("Resizing");
		run("Size...", "width="+size_X+" height="+size_Y+" constrain average interpolation=Bilinear"); // if you want to spatially downsample
	print("Splitting Stack" + stackTitle + " into " + num_chunks +" substacks");
	run("Stack Splitter", "number="+num_chunks);
	print("Done splitting");
	selectWindow(stackTitle); close();
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
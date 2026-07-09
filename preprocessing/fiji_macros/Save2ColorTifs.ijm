// macro to split 2 color image sequence; 
//input directory has subfolders of TSeries
// each TSeries folder has a series of images with Ch1_1:N first and Ch2_1:N second 
//splits into two substacks and saves each substack as a tif for each channel ; This only splits each TSeries evenly into 1 Tif for Ch1 and 1 Tif for Ch2! 
inputDir = getDirectory('Choose Input Directory');  //Choose input directory; with subfolders of all TSeries data
outputDir = getDirectory('Choose Output Directory'); //Choose an output directory to store subfolders with processed tifs

#@ Boolean (label="Perform resize") resize //Choose to resize 
#@ Integer (label="Size X") size_X
#@ Integer (label="Size Y") size_Y

listdir = getFileList(inputDir);
Array.print(listdir);

function split_stack(file0,n,outputDir,subdir) {
	run("Image Sequence...","open="+file0+" sort");
	if (resize)
		print("Resizing");
		run("Size...", "width="+size_X+" height="+size_Y+" constrain average interpolation=Bilinear"); // if you want to spatially downsample
	title=getTitle();
	run("Stack Splitter", "number="+n);
	File.makeDirectory(outputDir+subdir);
	selectWindow("stk_0001_"+title);
	saveAs("Tiff", outputDir+subdir+title+"_Ch1");
	print("Saved "+ outputDir+subdir+title+"_Ch1");
	selectWindow("stk_0002_"+title);
	saveAs("Tiff", outputDir+subdir+title+"_Ch2");
	print("Saved "+ outputDir+subdir+title+"_Ch2");
	run("Close All");
}

setBatchMode(true);
for(i=0; i< listdir.length; i++) {
	print("Processing " + inputDir+listdir[i]);
	subfiles=getFileList(inputDir+listdir[i]);
	tifs=newArray();
	for (j=0; j<subfiles.length;j++){
		if (endsWith(subfiles[j],'.ome.tif')) 
			tifs=Array.concat(tifs,inputDir+listdir[i]+subfiles[j]);
		};
	//print(tifs[0]);
	split_stack(tifs[0],2,outputDir,listdir[i]);
	print(i+1 +" of "+listdir.length);
}
print("Finished!");
setBatchMode(false);


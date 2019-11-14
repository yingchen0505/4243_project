### Sequence of Running:   

1. Run notebook 'crop.ipynb' to produce templates cropped from the training set.    
The templates are saved in the folder 'crop_train' under 'datasets'.   
   
2. Run the scripts 'find_waldo_method1.py' and 'find_waldo_method2.py'.    
The first script uses our first method (which downsizes larger test images) as outlined in the report.    
The outputs are saved in 'output1'.   
The second script uses our second method (which slices larger test images into smaller chunks) as outlined in the report.    
The outputs are saved in 'output2'.      
   
3. Run the notebook 'Concat.ipynb', which concatenates the two sets of output files under 'output1' and 'output2'.     
The concatenated output is written to 'unpruned_output'.    
   
4. Run the notebook 'RemoveDuplicates.ipynb', which removes some duplicate bounding boxes.     
The pruned output is written to 'output'.    
   
5. Run the notebook 'Evaluation.ipynb' to evaluate the output in the 'output' folder.      
      
	 
	  
------------

### Environment and Requirements:    

The conda environment is specified by 'environment.yml'.     
The packages required are indicated in the 'requirements.txt' requirement file.     

-----------

### Additional Information:    

The two PBS files are used to submit 'find_waldo_method1.py' and 'find_waldo_method2.py' to the PBS queues on NSCC HPC server.     
The current output folders 'output1' and 'output2' contain the output files obtained by running the two scipts on HPC using the given images.    
The images in the output folders are drawn in a way that the template on the left side of the graph is matched through green lines to the test image on the right side of the graph.    
The text files 'waldo.txt', 'wenda.txt', 'wizard.txt' are also written to the output folders by HPC.    
The 'find_waldo.o' files in the output folders are simply the print statements from the scripts for reference purpose only.     

The script 'remove_duplicates.py' is called by 'RemoveDuplicates.ipynb' to remove duplicate bounding boxes.     


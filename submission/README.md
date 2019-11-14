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
      
	  
	  
	  


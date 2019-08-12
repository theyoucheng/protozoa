
To run the script:  
--------------------
              python3 mobilenet-example.py



The format of output, for example, in 'ochiai.txt'
---------------------------------------------------

                    execution time       ranking time       execution time          ranking time 
                    (2000 mutants)       (2000 mutants)     (200 mutants)           (200 mutants)  
    10027 13534   122.50208592414856   3.015885591506958   10.677284479141235     1.4550728797912598


To plot:
---------
     cd outs
     python plot-mobilenet-bar.py

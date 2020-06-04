## PHASE PICKER ##

This is a simple deep learning model for automated phase picking.

There are two modules. The extract_window.py module extracts P, S and random noise from the IRIS network using Obspy. 
The model.py module contains a CNN that 'learns' how to differentiate noise from P and S waves. 

# Fusion_Grid_HemiSpheres.py
'''
This script reads the circle centers from a CSV file,
Creates spheres at each circle center in Fusion 360.

'''
import adsk.core, adsk.fusion, adsk.cam, traceback
import csv

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        # At the beginning of the `run` function to indicate the script is starting
        ui.messageBox('Script execution has started. Please wait...')

        design = app.activeProduct

        # Turn off history capture - this greatly speeds up the code but we lose the parametric editing ability after its built
        design.designType = adsk.fusion.DesignTypes.DirectDesignType 
        # Access the root component of the design
        rootComp = design.rootComponent

        # Path to your CSV file
        #r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"
        file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\combined_points.csv"


        # Read circle centers from the CSV file
        circle_centers = []
        with open(file_path, mode='r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)  # Skip the header row if there is one
            for row in csvreader:
                # Assuming the CSV columns are x, y, z, radius in order
                # exclude the first row of the csv
                circle = {'x': float(row[0]), 'y': float(row[1]), 'radius': float(row[2]), 'z': float(row[3]), }
                circle_centers.append(circle)

        # find the lowest z value and radius value of that circle
        extrude_z = min([circle['z'] for circle in circle_centers])
        #find the max z value
        max_z = max([circle['z'] for circle in circle_centers])

        #plane_z = min_z - 1

        # create a cylinder at each point in circle_centers
        for index, circle in enumerate(circle_centers):
            if circle['x'] < 20:
                columnName = f"Column_L_{index+1}"  # Generating a name based on the index
            else:
                columnName = f"Column_R_{index+1}"
            createCylinder(rootComp, circle['x'], circle['y'], circle['z'], circle['radius'], extrude_z, max_z, columnName)
                           
        
        print("Circle centers:", circle_centers)
        # # Create a hemisphere at each point in circle_centers
        # for circle in circle_centers:
        #     createHemisphere(rootComp, circle['x'], circle['y'], circle['z'], circle['radius'])

        # At the end of the `run` function to indicate completion
        ui.messageBox('Script execution completed successfully!')
        
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


def createCylinder(rootComp, x, y, z, radius, extrude_z, max_z, name):
    # Access the features collection of the root component
    features = rootComp.features

    # Create a new sketch on the xy plane
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    plane = max_z+0.2
    depth = extrude_z - max_z - 1 

    # Draw a circle in the sketch
    circles = sketch.sketchCurves.sketchCircles
    centerPoint = adsk.core.Point3D.create(x, y, plane)
    circle = circles.addByCenterRadius(centerPoint, radius)

    # extrude the circle two sides
    prof = sketch.profiles.item(0)
    extrudes = features.extrudeFeatures
    extrudeInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    distance = adsk.core.ValueInput.createByString(str(depth) + " cm")
    extrudeInput.setDistanceExtent(False, distance)
    extrudes.add(extrudeInput)

    # prof = sketch.profiles.item(0)
    # extrudes = features.extrudeFeatures
    # extrudeInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    # distance = adsk.core.ValueInput.createByString(str(-1) + " cm")
    # extrudeInput.setDistanceExtent(False, distance)
    # extrudes.add(extrudeInput)
    
    # Correctly access the last created body to assign a name
    # Assumes the last created body is the result of the revolve feature just added
    body = rootComp.bRepBodies.item(rootComp.bRepBodies.count - 1)
    body.name = name


def stop(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        ui.messageBox('Stop addin')
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

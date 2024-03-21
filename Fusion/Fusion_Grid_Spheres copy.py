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

        # # Define paths to your STL files
        # stl_file_path_right = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\delaunay_mesh_Right Side.stl"
        # stl_file_path_left = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\delaunay_mesh_Left Side.stl"

        # # Import the STL files
        # importSTLFile(rootComp, stl_file_path_right)
        # importSTLFile(rootComp, stl_file_path_left)

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
        print("Circle centers:", circle_centers)
        # Create a hemisphere at each point in circle_centers
        for index, circle in enumerate(circle_centers):
            if circle['x'] < 20:
                hemisphereName = f"Hemisphere_L_{index+1}"  # Generating a name based on the index
            else:
                hemisphereName = f"Hemisphere_R_{index+1}"
            createHemisphere(rootComp, circle['x'], circle['y'], circle['z'], circle['radius'], hemisphereName)


        # At the end of the `run` function to indicate completion
        ui.messageBox('Script execution completed successfully!')
        
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def createHemisphere(rootComp, x, y, z, radius, name):
    # Access the features collection of the root component
    features = rootComp.features

    # Create a new sketch on the xy plane
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    # Draw a circle in the sketch
    circles = sketch.sketchCurves.sketchCircles
    centerPoint = adsk.core.Point3D.create(x, y, z)
    circle = circles.addByCenterRadius(centerPoint, radius)

    # Draw a center line for the revolution
    lines = sketch.sketchCurves.sketchLines
    axis = lines.addByTwoPoints(adsk.core.Point3D.create(x, y - radius, z), adsk.core.Point3D.create(x, y + radius, z))

    # Revolve the half circle to create a hemisphere
    prof = sketch.profiles.item(0)
    revolves = features.revolveFeatures
    revolveInput = revolves.createInput(prof, axis, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    angle = adsk.core.ValueInput.createByString("360 deg")  # Revolve angle
    revolveInput.setAngleExtent(False, angle)
    revolveInput.isSolid = True
    revolves.add(revolveInput)

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

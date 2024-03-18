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
        # Access the root component of the design
        rootComp = design.rootComponent

        # Path to your CSV file
        #r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"
        file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"


        # Read circle centers from the CSV file
        circle_centers = []
        with open(file_path, mode='r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)  # Skip the header row if there is one
            for row in csvreader:
                # Assuming the CSV columns are x, y, z, radius in order
                circle = {'x': float(row[0]), 'y': float(row[1]), 'radius': float(row[2]), 'z': float(row[3]), }
                circle_centers.append(circle)
        print("Circle centers:", circle_centers)
        # Create a hemisphere at each point in circle_centers
        for circle in circle_centers:
            createHemisphere(rootComp, circle['x'], circle['y'], circle['z'], circle['radius'])

        # At the end of the `run` function to indicate completion
        ui.messageBox('Script execution completed successfully!')
        
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def createHemisphere(rootComp, x, y, z, radius):
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
    angle = adsk.core.ValueInput.createByString("180 deg")  # Revolve angle
    revolveInput.setAngleExtent(False, angle)
    revolveInput.isSolid = True
    revolves.add(revolveInput)

def stop(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        ui.messageBox('Stop addin')
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

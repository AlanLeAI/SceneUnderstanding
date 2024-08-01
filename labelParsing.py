import json
import csv
import re
import pandas as pd

with open('data/3DSSG/objects.json', 'r') as f:
    objects_data = json.load(f)

with open('data/3DSSG/relationships.json', 'r') as f:
    relationships_data = json.load(f)

with open('data/3DSSG/classes.txt') as f:
    classes = f.readlines()
    id_to_text = {}
    for i in range(len(classes)):
        classes[i] = classes[i].split("\t")
        id_to_text[classes[i][0]] = " ".join(classes[i][1:])

def describe_object(obj):
    """Generate a text description of an object."""
    label = obj['label']
    label_id = obj["global_id"]
    if label_id not in id_to_text.keys():
        return ""
    description = f"{label + ' is a ' + id_to_text[label_id]}"
    if 'attributes' in obj:
        attr_descriptions = []
        for key, values in obj['attributes'].items():
            if key == "state":
                attr_descriptions.append(f"{'The state of ' + label + ' is ' + ', '.join(values)}")
            elif key == "color":
                attr_descriptions.append(f"{'The color of ' +label + ' is ' + ', '.join(values)}")
            elif key == "shape":
                attr_descriptions.append(f"{'The shape of ' +label + ' is ' + ', '.join(values)}")
            elif key == "material":
                attr_descriptions.append(f"{'The material of ' +label + ' is ' + ', '.join(values)}")
            elif key == "texture":
                attr_descriptions.append(f"{'The texture of ' +label + ' is ' + ', '.join(values)}")
            elif key == "symmetry":
                attr_descriptions.append(f"{'The symmetry of ' +label + ' is ' + ', '.join(values)}")
            elif key == "other":
                attr_descriptions.append(f"{'Other attributes of ' +label + ' are ' + ', '.join(values)}")
        description += " " + ", ".join(attr_descriptions)
    if 'affordances' in obj:
        description += ". It can be used for " + ", ".join(obj['affordances'])
    return description

def process_data(objects_data, relationships_data):
    csv_data = []
    
    for scan in objects_data['scans']:
        scan_id = scan['scan']
        object_descriptions = []
        object_labels = []
        object_ids = []

        for obj in scan['objects']:
            desc = describe_object(obj)
            object_descriptions.append(desc)
            object_labels.append(obj['label'])
            object_ids.append(obj['id'])

        for rel_scan in relationships_data['scans']:
            if rel_scan['scan'] == scan_id:
                for rel in rel_scan['relationships']:
                    from_id, to_id, _, relation = rel
                    from_obj = next((o for o in scan['objects'] if o['id'] == str(from_id)), None)
                    to_obj = next((o for o in scan['objects'] if o['id'] == str(to_id)), None)
                    if from_obj and to_obj:
                        relation_desc = f"{from_obj['label']} is {relation} {to_obj['label']}"
                        object_descriptions.append(relation_desc)

        row = {
            'scan_id': scan_id,
            'descriptions': " | ".join(object_descriptions),
            'objects': ", ".join(object_labels),
            'object_ids': ", ".join(object_ids)
        }
        csv_data.append(row)

    with open('output.csv', 'w', newline='') as csvfile:
        fieldnames = ['scan_id', 'descriptions', 'objects', 'object_ids']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)


def process_description(description):
    # Split the description into sentences for each object
    sentences = [s.strip() for s in description.split('|')]
    # Create a dictionary to store details of each object
    object_details = {}
    for sentence in sentences:
        # Extract the name and details of the object
        name_match = re.search(r'^(\w+) is a (\w+)', sentence)
        if name_match:
            name = name_match.group(1)
            # Get the rest of the sentence for details
            details_start = sentence.find(name_match.group(0)) + len(name_match.group(0))
            details = sentence[details_start:].strip()
            object_details[name] = details
    return object_details

def summarize_objects(objects, descriptions):
    # Parse the description to get object details
    object_details = process_description(descriptions)
    # Split object list and iterate to create the summary
    objects_list = objects.split(', ')
    scene_description = "Objects in the scene are " + ', '.join(objects_list) + "."
    for obj in objects_list:
        if obj in object_details:
            # Append the description of each object to the scene description
            scene_description += f" The {obj} {object_details[obj]}."
    return scene_description


file_path = 'data/3DSSG/text_label.csv'
data = pd.read_csv(file_path)

# Apply the function to each row in the dataframe
data['scene_summary'] = data.apply(lambda row: summarize_objects(row['objects'], row['descriptions']), axis=1)
data = data[['scan_id', 'scene_summary']]
# Display the summarized scene descriptions
data.to_csv("text_preprocessed.csv")
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import pathfinding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from sklearn.cluster import DBSCAN
import traceback

# Set page configuration
st.set_page_config(
    page_title="School Evacuation Plan Generator",
    page_icon="🏫",
    layout="wide",
)

# Create directories to store temporary images
os.makedirs("temp", exist_ok=True)

# Define disaster types
DISASTER_TYPES = [
    "Earthquake",
    "Fire",
    "Flood",
    "Active Threat/Lockdown",
    "Tornado/Cyclone",
    "Chemical Spill"
]

# Color definitions
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
PURPLE = (128, 0, 128, 255)

class FloorPlanProcessor:
    def __init__(self):
        """Initialize the floor plan processor"""
        self.exit_keywords = ['exit', 'entrance', 'entry', 'door', 'emergency']
        self.room_keywords = ['room', 'office', 'class', 'library', 'lab', 'gym']
    
    def preprocess_image(self, image):
        """Preprocess the floor plan image for analysis with enhanced contrast handling"""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Check if image is too dark and enhance contrast if needed
        if np.mean(gray) < 100:  # If image is generally dark
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try different thresholding methods and pick the best one
        # Method 1: Adaptive thresholding
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Method 2: Otsu's thresholding
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Choose the method that gives clearer lines (more white pixels but not too many)
        white_pixels_adaptive = np.sum(binary_adaptive == 255)
        white_pixels_otsu = np.sum(binary_otsu == 255)
        total_pixels = binary_adaptive.size
        
        # Select the binary image with better white pixel ratio (between 5-40% of image)
        if 0.05 * total_pixels <= white_pixels_adaptive <= 0.4 * total_pixels:
            binary = binary_adaptive
        else:
            binary = binary_otsu
        
        # Perform morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def detect_exits(self, image, binary_image):
        """Detect exits in the floor plan with improved accuracy"""
        exits = []
        
        # Method 1: Look for green areas (exits are often marked in green)
        if len(image.shape) == 3:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define range for green color (broader range)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # Create mask for green areas
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours in the green mask
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 30:  # Lower threshold to catch smaller exit markers
                    # Get the centroid of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Add this as an exit
                        exits.append({
                            'id': len(exits),
                            'name': f'Exit {len(exits)+1}',
                            'position': (cx, cy),
                            'detected_by': 'color'
                        })
        
        # Method 2: Find potential exits along the perimeter
        # Get image dimensions
        h, w = binary_image.shape
        
        # Find the contours of the building
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (presumably the building outline)
            building_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding box of the building
            x, y, w_box, h_box = cv2.boundingRect(building_contour)
            
            # Check for discontinuities in the contour that might indicate doorways
            epsilon = 0.005 * cv2.arcLength(building_contour, True)  # More precise approximation
            approx = cv2.approxPolyDP(building_contour, epsilon, True)
            
            # Look for potential doorways - places where the contour has sharp turns
            perimeter_points = []
            
            for i in range(len(approx)):
                p1 = approx[i][0]
                p2 = approx[(i+1) % len(approx)][0]
                
                # Calculate distance between consecutive points
                dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                # If points are far apart, this might be a doorway or passage
                if dist > 20:
                    # Check if this point is on or near the perimeter
                    is_perimeter = (
                        abs(p1[0] - x) < 20 or abs(p1[0] - (x + w_box)) < 20 or 
                        abs(p1[1] - y) < 20 or abs(p1[1] - (y + h_box)) < 20
                    )
                    
                    if is_perimeter:
                        perimeter_points.append(p1)
            
            # Add perimeter points as potential exits
            for point in perimeter_points:
                # Check if this point is too close to existing exits
                is_duplicate = False
                for existing_exit in exits:
                    ex, ey = existing_exit['position']
                    if np.sqrt((point[0] - ex)**2 + (point[1] - ey)**2) < 50:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    exits.append({
                        'id': len(exits),
                        'name': f'Exit {len(exits)+1}',
                        'position': (point[0], point[1]),
                        'detected_by': 'perimeter'
                    })
        
        # Method 3: Strategic fallback exits if few or no exits detected
        if len(exits) < 2:
            # Place exits at strategic locations around the perimeter
            border_margin = 15
            strategic_positions = [
                {'pos': (w//2, border_margin), 'name': 'North Exit'},
                {'pos': (w//2, h-border_margin), 'name': 'South Exit'},
                {'pos': (border_margin, h//2), 'name': 'West Exit'},
                {'pos': (w-border_margin, h//2), 'name': 'East Exit'}
            ]
            
            for pos_info in strategic_positions:
                is_duplicate = False
                for existing_exit in exits:
                    ex, ey = existing_exit['position']
                    sp_x, sp_y = pos_info['pos']
                    if np.sqrt((sp_x - ex)**2 + (sp_y - ey)**2) < w//6:  # Don't place too close to existing
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    exits.append({
                        'id': len(exits),
                        'name': pos_info['name'],
                        'position': pos_info['pos'],
                        'detected_by': 'strategic'
                    })
        
        # Rename exits based on their position and ensure Main Exit
        if exits:
            for i, exit_point in enumerate(exits):
                x, y = exit_point['position']
                
                # Determine exit type based on position
                if y < h//4:  # Top
                    prefix = "North"
                elif y > 3*h//4:  # Bottom
                    prefix = "South"
                else:
                    if x < w//4:  # Left
                        prefix = "West"
                    elif x > 3*w//4:  # Right
                        prefix = "East"
                    else:
                        prefix = "Central"
                
                exits[i]['name'] = f"{prefix} Exit {i+1}"
            
            # Designate main exit (typically the one nearest the bottom center)
            main_exit_score = float('inf')
            main_exit_idx = 0
            
            for i, exit_point in enumerate(exits):
                x, y = exit_point['position']
                # Score based on distance from bottom center
                score = abs(x - w//2) + abs(y - h) * 0.5
                if score < main_exit_score:
                    main_exit_score = score
                    main_exit_idx = i
            
            exits[main_exit_idx]['name'] = "Main Exit"
        
        return exits
    
    def detect_rooms(self, binary_image):
        """Detect rooms in the floor plan with improved error handling"""
        # Find contours in the binary image
        try:
            contours, hierarchy = cv2.findContours(
                binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            
            rooms = []
            min_room_area = 500  # Minimum area to be considered a room
            
            if hierarchy is not None and len(hierarchy) > 0 and len(hierarchy[0]) > 0:
                hierarchy = hierarchy[0]
                
                for i, contour in enumerate(contours):
                    # Avoid index errors by checking hierarchy bounds
                    if i >= len(hierarchy):
                        continue
                        
                    # Check if this is an inner contour (potential room)
                    if hierarchy[i][3] != -1:  # Has parent contour
                        area = cv2.contourArea(contour)
                        
                        if area > min_room_area:
                            # Get bounding rectangle
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Check aspect ratio to filter out very narrow spaces
                            aspect_ratio = float(w) / h if h > 0 else 0
                            if 0.2 < aspect_ratio < 5:  # Reasonable room aspect ratio
                                rooms.append({
                                    'id': len(rooms),
                                    'x': x,
                                    'y': y,
                                    'width': w,
                                    'height': h,
                                    'center': (x + w//2, y + h//2)
                                })
        except Exception as e:
            print(f"Error in detect_rooms: {str(e)}")
            # Fall back to simple contour detection
            
        # If no rooms detected with hierarchy, try simple external contours
        if not rooms:
            try:
                contours, _ = cv2.findContours(
                    binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > min_room_area:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check aspect ratio
                        aspect_ratio = float(w) / h if h > 0 else 0
                        if 0.2 < aspect_ratio < 5:
                            rooms.append({
                                'id': len(rooms),
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'center': (x + w//2, y + h//2)
                            })
            except Exception as e:
                print(f"Error in detect_rooms fallback: {str(e)}")
        
        # If still no rooms, create a grid of rooms
        if not rooms:
            h, w = binary_image.shape
            grid_size = 4  # 4x4 grid
            cell_w = w // grid_size
            cell_h = h // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x = j * cell_w
                    y = i * cell_h
                    
                    rooms.append({
                        'id': i * grid_size + j,
                        'x': x,
                        'y': y,
                        'width': cell_w,
                        'height': cell_h,
                        'center': (x + cell_w//2, y + cell_h//2)
                    })
        
        return rooms
    
    def create_grid(self, binary_image, rooms): # Add rooms as parameter
        """Create a grid representation of the floor plan for pathfinding"""
        # Invert the binary image for pathfinding (walls=0, free space=1)
        walkable_grid = (binary_image == 0).astype(int)

        # Explicitly mark rooms as non-walkable
        for room in rooms:
            x, y, w, h = room['x'], room['y'], room['width'], room['height']
            # Ensure bounds are within the image dimensions
            x2, y2 = min(x + w, walkable_grid.shape[1]), min(y + h, walkable_grid.shape[0])
            walkable_grid[y:y2, x:x2] = 0 # Mark the room area as non-walkable

        # Perform morphological operations to find corridors (apply AFTER marking rooms)
        kernel = np.ones((15, 15), np.uint8)  # Larger kernel to identify main paths
        corridors = cv2.morphologyEx(walkable_grid.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Enhance corridors
        walkable_grid = np.maximum(walkable_grid, corridors)

        # ... rest of your scaling code ...
        # Scale down the grid for efficiency
        h, w = walkable_grid.shape
        grid_scale = max(1, min(w, h) // 100)  # Adaptive scaling based on image size

        scaled_h, scaled_w = h // grid_scale, w // grid_scale
        grid_data = np.zeros((scaled_h, scaled_w))

        for i in range(scaled_h):
            for j in range(scaled_w):
                # Check the corresponding region in the original image
                region = walkable_grid[
                    i*grid_scale:min((i+1)*grid_scale, h),
                    j*grid_scale:min((j+1)*grid_scale, w)
                ]

                # If more than 50% of pixels are walkable, mark cell as walkable
                if np.mean(region) > 0.5:
                    grid_data[i, j] = 1

        return grid_data, grid_scale
    
    def calculate_evacuation_routes(self, grid_data, rooms, exits, grid_scale):
        """Calculate evacuation routes from each room to the nearest exit with corridor preference"""
        pathfinder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        grid = Grid(matrix=grid_data.tolist())
        
        # Enhance corridor preference by adding weight to non-corridor cells
        h, w = grid_data.shape
        for y in range(h):
            for x in range(w):
                node = grid.node(x, y)
                # If walkable but likely not a corridor (based on surrounding cells)
                if grid_data[y, x] == 1:
                    surrounding_sum = 0
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                surrounding_sum += grid_data[ny, nx]
                                count += 1
                    
                    # If surrounded by many walls, it's likely not a good path
                    corridor_score = surrounding_sum / count if count > 0 else 0
                    if corridor_score < 0.6:  # Not a good corridor
                        node.weight = 5  # Make this path less preferred
        
        routes = {}
        
        for room in rooms:
            # Scale the room center to match the grid
            start_x = room['center'][0] // grid_scale
            start_y = room['center'][1] // grid_scale
            
            # Ensure within grid bounds
            start_x = max(0, min(start_x, w-1))
            start_y = max(0, min(start_y, h-1))
            
            # Check if the starting point is walkable, if not find nearby walkable point
            if grid_data[start_y, start_x] == 0:
                # Look for a nearby walkable cell with expanding search
                found = False
                for search_radius in range(1, 15):  # Search larger radius
                    for dy in range(-search_radius, search_radius+1):
                        for dx in range(-search_radius, search_radius+1):
                            # Only check points at the current search perimeter
                            if abs(dx) + abs(dy) == search_radius:
                                ny, nx = start_y + dy, start_x + dx
                                if (0 <= ny < h and 0 <= nx < w and 
                                    grid_data[ny, nx] == 1):
                                    start_x, start_y = nx, ny
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                
                # If still no walkable cell found, skip this room
                if not found:
                    continue
            
            start = grid.node(start_x, start_y)
            
            # Find the nearest exit
            nearest_exit = None
            shortest_path = None
            min_length = float('inf')
            
            for exit_point in exits:
                # Scale the exit position to match the grid
                end_x = exit_point['position'][0] // grid_scale
                end_y = exit_point['position'][1] // grid_scale
                
                # Ensure within grid bounds
                end_x = max(0, min(end_x, w-1))
                end_y = max(0, min(end_y, h-1))
                
                # Ensure exit is walkable
                if grid_data[end_y, end_x] == 0:
                    # Look for nearby walkable cell with expanding search
                    found = False
                    for search_radius in range(1, 15):  # Search larger radius
                        for dy in range(-search_radius, search_radius+1):
                            for dx in range(-search_radius, search_radius+1):
                                # Only check points at the current search perimeter
                                if abs(dx) + abs(dy) == search_radius:
                                    ny, nx = end_y + dy, end_x + dx
                                    if (0 <= ny < h and 0 <= nx < w and 
                                        grid_data[ny, nx] == 1):
                                        end_x, end_y = nx, ny
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    
                    if not found:
                        continue
                
                end = grid.node(end_x, end_y)
                
                # Reset the grid for new path calculation
                grid.cleanup()
                
                # Find path
                path, runs = pathfinder.find_path(start, end, grid)
                
                # Weight the path by actual length and exit importance
                # Prefer main exits for rooms closer to them
                path_weight = len(path)
                if "Main" in exit_point['name'] and path_weight < 100:
                    path_weight *= 0.8  # Prefer main exits for nearby rooms
                
                if path and path_weight < min_length:
                    min_length = path_weight
                    shortest_path = path
                    nearest_exit = exit_point
            
            if shortest_path:
                # Scale back to original image size
                full_path = []
                # Smooth the path by taking fewer points
                step = max(1, len(shortest_path) // 20)  # Take at most 20 points along the path
                for i in range(0, len(shortest_path), step):
                    node = shortest_path[i]
                    x = node.x * grid_scale
                    y = node.y * grid_scale
                    full_path.append((x, y))
                
                # Always include the end point
                if shortest_path:
                    last_node = shortest_path[-1]
                    x = last_node.x * grid_scale
                    y = last_node.y * grid_scale
                    if full_path[-1] != (x, y):
                        full_path.append((x, y))
                
                routes[room['id']] = {
                    'room': room,
                    'exit': nearest_exit,
                    'path': full_path
                }
        
        return routes
    
    def generate_evacuation_map(self, original_image, rooms, exits, routes, disaster_type):
        """Generate a clearer visual evacuation map with routes marked"""
        try:
            # Ensure image is in color (3 channels) for consistent processing
            if len(original_image.shape) == 2: # If it's grayscale (2D)
                img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB) # Convert to RGB
            else:
                img = original_image.copy() # Otherwise, just copy the original

            # Now, the rest of your code can assume 'img' is 3 channels or more
            # Check if image is too dark (low average pixel value) - apply to RGB
            if len(img.shape) == 3: # Ensure it's 3 channels (which it should be now)
                avg_brightness = np.mean(img)
                if avg_brightness < 80:  # Image is very dark
                    # Convert to grayscale, invert, then back to RGB
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.cvtColor(cv2.bitwise_not(gray), cv2.COLOR_GRAY2RGB)

            # Convert to RGBA if it's not already
            # This block should now work reliably as img is at least 3 channels
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgba = np.ones((img.shape[0], img.shape[1], 4), dtype=np.uint8) * 255
                img_rgba[:, :, 0:3] = img
                img_rgba[:, :, 3] = 255
                img = img_rgba

            # Brighten the image slightly to make annotations more visible
            # This should also be safe now that img is at least 3 channels
            img_brightened = np.minimum(img[:, :, :3] * 1.2, 255).astype(np.uint8)

            # ... rest of your generate_evacuation_map function ...
            # Ensure consistent 4-channel output
            if img_brightened.shape[2] == 3:
                output_img = np.ones((img_brightened.shape[0], img_brightened.shape[1], 4), dtype=np.uint8) * 255
                output_img[:, :, :3] = img_brightened
            else:
                output_img = img_brightened

            # Convert to PIL Image for drawing
            img_pil = Image.fromarray(output_img)
            draw = ImageDraw.Draw(img_pil)

            # Set font and marker sizes based on image dimensions - REDUCED sizes
            img_size = max(img.shape[0], img.shape[1])
            font_size = max(10, min(16, img_size // 60))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                title_font = ImageFont.truetype("arial.ttf", int(font_size * 1.5))
            except:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # REDUCED line width for better clarity
            line_width = max(1, img_size // 500)
            marker_size = max(4, img_size // 120)
            
            # Draw rooms with outlines - thinner lines
            for room in rooms:
                draw.rectangle(
                    [(room['x'], room['y']), 
                    (room['x'] + room['width'], room['y'] + room['height'])],
                    outline=BLUE,
                    width=max(1, line_width-1)
                )
                # Label the room
                label = f"R {room['id']+1}"
                draw.text(
                    (room['center'][0]-10, room['center'][1]-10), 
                    label, 
                    fill=BLUE, 
                    font=font
                )
            
            # Define colors for different evacuation routes - use more distinct colors
            colors = [
                (255, 0, 0, 255),      # Red
                (128, 0, 128, 255),    # Purple
                (0, 128, 0, 255),      # Dark Green
                (255, 165, 0, 255),    # Orange
                (0, 128, 128, 255),    # Teal
                (0, 0, 255, 255)       # Blue
            ]
            
            # Group routes by exit for better color coding
            routes_by_exit = {}
            for room_id, route_info in routes.items():
                exit_id = route_info['exit']['id']
                if exit_id not in routes_by_exit:
                    routes_by_exit[exit_id] = []
                routes_by_exit[exit_id].append(route_info)
            
            # Draw evacuation routes with different colors for different exits
            # THINNER LINES for better clarity
            for exit_id, exit_routes in routes_by_exit.items():
                color = colors[exit_id % len(colors)]
                
                for route_info in exit_routes:
                    path = route_info['path']
                    
                    # Draw the path with the color for this exit - REDUCED thickness
                    for i in range(len(path) - 1):
                        start_point = path[i]
                        end_point = path[i + 1]
                        draw.line([start_point, end_point], fill=color, width=line_width)
                    
                    # Draw fewer arrows along the path to reduce clutter
                    if len(path) > 8:
                        arrow_spacing = max(len(path) // 4, 5)  # Fewer arrows
                        for i in range(0, len(path)-1, arrow_spacing):
                            start = path[i]
                            end = path[i+1]
                            # Calculate angle for arrow
                            angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                            # Draw arrow - SMALLER
                            arrow_length = marker_size * 1.5  # Reduced size
                            arrow_width = marker_size * 0.8   # Reduced size
                            arrow_x = start[0] + arrow_length * np.cos(angle)
                            arrow_y = start[1] + arrow_length * np.sin(angle)
                            
                            # Draw arrowhead - thinner line
                            draw.line([start, (arrow_x, arrow_y)], fill=color, width=line_width)
                            draw.polygon([
                                (arrow_x, arrow_y),
                                (arrow_x - arrow_width * np.cos(angle - np.pi/6), arrow_y - arrow_width * np.sin(angle - np.pi/6)),
                                (arrow_x - arrow_width * np.cos(angle + np.pi/6), arrow_y - arrow_width * np.sin(angle + np.pi/6))
                            ], fill=color)
            
            # Draw exits with SMALLER green markers
            for exit_point in exits:
                pos = exit_point['position']
                
                # Draw a green circle for each exit - smaller
                exit_size = marker_size * 1.5  # Reduced size
                draw.ellipse(
                    [(pos[0]-exit_size, pos[1]-exit_size), 
                    (pos[0]+exit_size, pos[1]+exit_size)], 
                    fill=GREEN
                )
                
                # Label the exit with smaller font
                draw.text(
                    (pos[0]-exit_size, pos[1]-exit_size*2), 
                    exit_point['name'], 
                    fill=GREEN, 
                    font=font
                )
            
            # Add disaster type label and title with larger font
            # Make background semi-transparent for better readability
            draw.rectangle(
                [(10, 10), (400, 40)],
                fill=(255, 255, 255, 200)
            )
            draw.text(
                (20, 15), 
                f"Evacuation Plan for: {disaster_type}", 
                fill=(0, 0, 0, 255), 
                font=title_font
            )
            
            
            # Convert back to numpy array
            return np.array(img_pil)
        
        except Exception as e:
            st.error(f"Error generating evacuation map: {str(e)}")
            traceback.print_exc()
            
            # Create a blank white image as fallback with same dimensions
            h, w = original_image.shape[:2]
            fallback_img = np.ones((h, w, 4), dtype=np.uint8) * 255
            
            # Add error message to the image
            fallback_pil = Image.fromarray(fallback_img)
            draw = ImageDraw.Draw(fallback_pil)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            draw.text((20, 20), f"Error processing image: {str(e)}", fill=(255, 0, 0, 255), font=font)
            draw.text((20, 50), "Try a different image or adjust the image quality", fill=(0, 0, 0, 255), font=font)
            
            # Convert back to numpy array
            return np.array(fallback_pil)   
        
    def generate_text_instructions(self, rooms, routes, disaster_type):
        """Generate text-based evacuation instructions"""
        instructions = {}
        
        # Group rooms by exit
        rooms_by_exit = {}
        for room_id, route_info in routes.items():
            exit_name = route_info['exit']['name']
            if exit_name not in rooms_by_exit:
                rooms_by_exit[exit_name] = []
            rooms_by_exit[exit_name].append(room_id)
        
        # Generate instructions for each exit group
        for exit_name, room_ids in rooms_by_exit.items():
            room_numbers = [f"Room {room_id+1}" for room_id in room_ids]
            room_text = ", ".join(room_numbers)
            instructions[exit_name] = {
                'rooms': room_ids,
                'text': f"For {disaster_type}: {room_text} should evacuate through {exit_name}."
            }
        
        return instructions


# Function to generate disaster-specific guidance
def generate_disaster_guidance(disaster_type):
    """Generate disaster-specific guidance for different disaster types"""
    guidance = {
        "Earthquake": "DROP to the ground, COVER under a sturdy desk or table, and HOLD ON until the shaking stops. Stay away from windows. Follow evacuation routes once safe to move.",
        "Fire": "If you detect fire or smoke, activate the nearest fire alarm. Stay low to avoid smoke inhalation. Feel doors before opening - if hot, find another exit route. Follow the marked evacuation paths.",
        "Flood": "Move to higher floors if inside. Never walk through moving water. Follow designated evacuation routes to higher ground.",
        "Active Threat/Lockdown": "Run if you can escape safely. Hide if evacuation isn't possible - lock doors, turn off lights, silence phones. Fight only as a last resort.",
        "Tornado/Cyclone": "Move to interior rooms on the lowest floor. Stay away from windows. Cover your head and neck. Follow staff instructions for moving to designated shelter areas.",
        "Chemical Spill": "If indoors, shelter in place unless directed to evacuate. Close all windows and doors. If evacuating, move upwind of the spill area."
    }
    
    return guidance.get(disaster_type, "Follow the evacuation routes indicated on the map. Walk, don't run. Stay calm and help others if safe to do so.")

def use_custom_sample_image(custom_path=None):
    """Uses a provided image path or creates a sample floor plan"""
    if custom_path and os.path.exists(custom_path):
        try:
            # Try to load the custom image
            img = cv2.imread(custom_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            st.error(f"Error loading custom sample image: {str(e)}")
    
    # Hard-coded path to your sample floor plan
    default_sample_path = r"sample_map.jpg"  # Update this path
    if os.path.exists(default_sample_path):
        try:
            img = cv2.imread(default_sample_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            st.error(f"Error loading default sample image: {str(e)}")
    
    # Create a simple sample as fallback - you could keep a simplified version
    # of your existing code here or return a blank image with text
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    # Add text indicating sample image couldn't be loaded
    cv2.putText(img, "Sample floor plan not available", (400, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


# Main application function
def main():
    st.title("School Evacuation Plan Generator")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Floor Plan")
        uploaded_file = st.file_uploader("Choose a floor plan image", type=["jpg", "jpeg", "png"])
        
        st.header("Disaster Type")
        disaster_type = st.selectbox("Select disaster scenario", DISASTER_TYPES)
        
        if uploaded_file is not None:
            process_button = st.button("Generate Evacuation Plan")
        
        # Sample floor plan options
        st.header("Don't have a floor plan?")
        use_default_sample = st.button("Use A Sample Floor Plan")

    
    # Main content area
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Process when button is clicked
        if process_button:
            with st.spinner("Processing floor plan..."):
                try:
                    # Process the image
                    processor = FloorPlanProcessor()
                    
                    # Preprocess image
                    binary_image = processor.preprocess_image(img_array)
                    
                    # Detect rooms and exits
                    rooms = processor.detect_rooms(binary_image)
                    exits = processor.detect_exits(img_array, binary_image)
                    
                    # Create grid for pathfinding
                    grid_data, grid_scale = processor.create_grid(binary_image, rooms)
                    
                    # Calculate evacuation routes
                    routes = processor.calculate_evacuation_routes(grid_data, rooms, exits, grid_scale)
                    
                    # Generate evacuation map
                    # Generate evacuation map with error handling
                    try:
                        evacuation_map = processor.generate_evacuation_map(img_array, rooms, exits, routes, disaster_type)
                    except Exception as e:
                        st.error(f"Error generating evacuation map: {str(e)}")
                        st.write("Falling back to original floor plan.")
                        evacuation_map = img_array                    
                
                    # Generate text instructions
                    instructions = processor.generate_text_instructions(rooms, routes, disaster_type)
                    
                    # Generate disaster-specific guidance
                    guidance = generate_disaster_guidance(disaster_type)
                    
                    # Display results
                    st.header(f"Evacuation Plan for {disaster_type}")
                    
                    # Display the evacuation map
                    st.subheader("Visual Evacuation Map")
                    st.image(evacuation_map, caption=f"{disaster_type} Evacuation Plan", use_container_width=True)
                    
                    # Display text instructions
                    st.subheader("Evacuation Instructions")
                    for exit_name, exit_info in instructions.items():
                        st.markdown(f"**{exit_info['text']}**")
                    
                    # Display disaster-specific guidance
                    st.subheader("Disaster-Specific Guidance")
                    st.write(guidance)
                    
# Download options
                    map_path = os.path.join("temp", "evacuation_map.png")
                    cv2.imwrite(map_path, cv2.cvtColor(evacuation_map, cv2.COLOR_RGB2BGR))
                    
                    with open(map_path, "rb") as file:
                        st.download_button(
                            label="Download Evacuation Map",
                            data=file,
                            file_name=f"{disaster_type}_evacuation_map.png",
                            mime="image/png"
                        )
                    
                    # Generate PDF report
                    st.subheader("Download Full Report")
                    
                    # Save instructions to text file for download
                    instructions_text = f"EVACUATION PLAN FOR: {disaster_type}\n\n"
                    instructions_text += "EVACUATION INSTRUCTIONS:\n"
                    for exit_name, exit_info in instructions.items():
                        instructions_text += f"- {exit_info['text']}\n"
                    
                    instructions_text += f"\nDISASTER-SPECIFIC GUIDANCE:\n{guidance}"
                    
                    instructions_path = os.path.join("temp", "evacuation_instructions.txt")
                    with open(instructions_path, "w") as f:
                        f.write(instructions_text)
                    
                    with open(instructions_path, "r") as file:
                        st.download_button(
                            label="Download Instructions",
                            data=file,
                            file_name=f"{disaster_type}_evacuation_instructions.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"Error processing floor plan: {str(e)}")
    
    elif use_default_sample:
        # Generate and display sample floor plan
        with st.spinner("Generating sample floor plan and evacuation plan..."):
            try:
                # Use the sample floor plan image
                sample_img = use_custom_sample_image()  # This function will use the default sample
                
                # Continue with existing processing...
                processor = FloorPlanProcessor()
                
                # Preprocess image
                binary_image = processor.preprocess_image(sample_img)
                
                # Detect rooms and exits
                rooms = processor.detect_rooms(binary_image)
                exits = processor.detect_exits(sample_img, binary_image)
                
                # Create grid for pathfinding
                grid_data, grid_scale = processor.create_grid(binary_image, rooms)
                
                # Calculate evacuation routes
                routes = processor.calculate_evacuation_routes(grid_data, rooms, exits, grid_scale)
                
                # Generate evacuation map
                evacuation_map = processor.generate_evacuation_map(sample_img, rooms, exits, routes, disaster_type)
                
                # Generate text instructions
                instructions = processor.generate_text_instructions(rooms, routes, disaster_type)
                
                # Generate disaster-specific guidance
                guidance = generate_disaster_guidance(disaster_type)
                
                # Display original sample floor plan
                st.header("Sample School Floor Plan")
                st.image(sample_img, caption="Sample School Floor Plan", use_container_width=True)
                
                # Display results
                st.header(f"Sample Evacuation Plan for {disaster_type}")
                
                # Display the evacuation map
                st.subheader("Visual Evacuation Map")
                st.image(evacuation_map, caption=f"{disaster_type} Evacuation Plan", use_container_width=True)
                
                # Display text instructions
                st.subheader("Evacuation Instructions")
                for exit_name, exit_info in instructions.items():
                    st.markdown(f"**{exit_info['text']}**")
                
                # Display disaster-specific guidance
                st.subheader("Disaster-Specific Guidance")
                st.write(guidance)
                
                # Download options
                map_path = os.path.join("temp", "sample_evacuation_map.png")
                cv2.imwrite(map_path, cv2.cvtColor(evacuation_map, cv2.COLOR_RGBA2BGR))
                
                with open(map_path, "rb") as file:
                    st.download_button(
                        label="Download Sample Evacuation Map",
                        data=file,
                        file_name=f"sample_{disaster_type}_evacuation_map.png",
                        mime="image/png"
                    )
                
                # Save instructions to text file for download
                instructions_text = f"SAMPLE EVACUATION PLAN FOR: {disaster_type}\n\n"
                instructions_text += "EVACUATION INSTRUCTIONS:\n"
                for exit_name, exit_info in instructions.items():
                    instructions_text += f"- {exit_info['text']}\n"
                
                instructions_text += f"\nDISASTER-SPECIFIC GUIDANCE:\n{guidance}"
                
                instructions_path = os.path.join("temp", "sample_evacuation_instructions.txt")
                with open(instructions_path, "w") as f:
                    f.write(instructions_text)
                
                with open(instructions_path, "r") as file:
                    st.download_button(
                        label="Download Sample Instructions",
                        data=file,
                        file_name=f"sample_{disaster_type}_evacuation_instructions.txt",
                        mime="text/plain"
                    )
            
            except Exception as e:
                st.error(f"Error generating sample plan: {str(e)}")
    
    else:
        # Display app description when no file is uploaded
        st.header("Welcome to the School Evacuation Plan Generator")
        st.write("""
        This application helps school administrators quickly generate evacuation plans for various disaster scenarios.
        
        ### How to use:
        1. Upload your school's floor plan image using the sidebar
        2. Select the disaster type from the dropdown menu
        3. Click "Generate Evacuation Plan"
        4. View and download your custom evacuation map and instructions
        
        ### Features:
        - Automatic identification of rooms and exits
        - Optimal evacuation route calculation
        - Disaster-specific evacuation guidance
        - Downloadable evacuation maps and instructions
        
        Don't have a floor plan? Try the "Use Sample Floor Plan" button to see how it works!
        """)


if __name__ == "__main__":
    main()

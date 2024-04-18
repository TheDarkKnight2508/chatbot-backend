import aiohttp
import asyncio
import time
from collections import OrderedDict
import openai

api_key = "sk-proj-ll3cVDwKOGa0Frh0vsFET3BlbkFJy4f2Pq3nB7uav3mlKkvb"
openai.api_key = api_key

# Function to fetch data from Google Maps API with advanced caching
class LRUCache:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # Move the accessed key to the end to mark it as most recently used
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        if key in self.cache:
            # If key already exists, move it to the end to mark it as most recently used
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # If the cache is full, remove the least recently used item (first item)
            self.cache.popitem(last=False)
        self.cache[key] = value

async def fetch_api_data(source_lat, source_long, dest_lat, dest_long, cache=None):
    if cache is None:
        cache = LRUCache()
    cache_key = f"{source_lat},{source_long},{dest_lat},{dest_long}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    directions_url = f"https://www.google.com/maps/dir/{source_lat},{source_long}/{dest_lat},{dest_long}"
    map_url = f"https://google.com/maps?q={dest_lat},{dest_long}"
    async with aiohttp.ClientSession() as session:
        async with session.get(directions_url) as directions_response, session.get(map_url) as map_response:
            if directions_response.status == 200 and map_response.status == 200:
                directions_html = await directions_response.text()
                map_html = await map_response.text()
                cache.put(cache_key, (directions_html, map_html))
                return directions_html, map_html
            else:
                print("Error fetching data from Google Maps API")
                return None, None

# Function to retrieve distance between two points using Google Maps API
async def get_distance(source_lat, source_long, dest_lat, dest_long):
    directions_html, _ = await fetch_api_data(source_lat, source_long, dest_lat, dest_long)
    if directions_html:
        # Parse the distance from directions_html
        distance = "10 miles"  # Placeholder distance
        return distance
    else:
        print("Failed to fetch directions. Unable to calculate distance.")
        return None
    
async def generate_training_data(source_lat, source_long, dest_lat, dest_long):
    directions_html, _ = await fetch_api_data(source_lat, source_long, dest_lat, dest_long)
    if directions_html:
        directions = parse_directions(directions_html)
        if directions:
            # Construct training examples based on the parsed directions, time, and distance
            train_data = [
                {"input": f"How do I get from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Directions: {directions}"},
                {"input": f"What is the route from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Directions: {directions}"},
                {"input": f"I need directions from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Directions: {directions}"},
                {"input": f"Show me the way from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Directions: {directions}"},
                {"input": f"Navigate from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Directions: {directions}"},
                {"input": f"Can you tell me how to go from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Directions: {directions}"},
                {"input": f"Directions to {dest_lat},{dest_long} from {source_lat},{source_long}.", "output": f"Directions: {directions}"},
                {"input": f"Guide me from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Directions: {directions}"},
                {"input": f"What's the way from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Directions: {directions}"},
                {"input": f"Please show the route from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Directions: {directions}"},
                {"input": f"I'm at {source_lat},{source_long}. How do I reach {dest_lat},{dest_long}?", "output": f"Directions: {directions}"},
                {"input": f"How far is {dest_lat},{dest_long} from {source_lat},{source_long}?", "output": f"Distance: 10 miles"},
                {"input": f"What is the distance between {source_lat},{source_long} and {dest_lat},{dest_long}?", "output": f"Distance: 10 miles"},
                {"input": f"How long will it take to travel from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Time: 30 minutes"},
                {"input": f"Can you provide the estimated time to reach {dest_lat},{dest_long} from {source_lat},{source_long}?", "output": f"Time: 30 minutes"},
                {"input": f"Show me the time required to travel from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Time: 30 minutes"},
                {"input": f"I'm starting at {source_lat},{source_long}. How much time will it take to get to {dest_lat},{dest_long}?", "output": f"Time: 30 minutes"},
                {"input": f"Tell me the duration to reach {dest_lat},{dest_long} from {source_lat},{source_long}.", "output": f"Time: 30 minutes"},
                {"input": f"Provide the time taken to travel from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Time: 30 minutes"},
                {"input": f"How long is the journey from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Time: 30 minutes"},
                {"input": f"Distance between {source_lat},{source_long} and {dest_lat},{dest_long}.", "output": f"Distance: 10 miles"},
                {"input": f"What is the time to travel from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Time: 30 minutes"},
                {"input": f"Tell me the distance to {dest_lat},{dest_long} from {source_lat},{source_long}.", "output": f"Distance: 10 miles"},
                {"input": f"How much time does it take to reach {dest_lat},{dest_long} from {source_lat},{source_long}?", "output": f"Time: 30 minutes"},
                {"input": f"Distance and time from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"Show the distance and time from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"I'm planning a trip from {source_lat},{source_long} to {dest_lat},{dest_long}. How far is it and how long will it take?", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"Tell me both the distance and time to reach {dest_lat},{dest_long} from {source_lat},{source_long}.", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"Distance and estimated time from {source_lat},{source_long} to {dest_lat},{dest_long}.", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"What's the distance and time required to travel from {source_lat},{source_long} to {dest_lat},{dest_long}?", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"I want to know the distance and time to {dest_lat},{dest_long} from {source_lat},{source_long}.", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"Provide information about distance and time between {source_lat},{source_long} and {dest_lat},{dest_long}.", "output": f"Distance: 10 miles, Time: 30 minutes"},
                {"input": f"How much distance do I need to travel from {source_lat},{source_long} to {dest_lat},{dest_long} and how long will it take?", "output": f"Distance: 10 miles, Time: 30 minutes"},
            ]
            return train_data
    return []


    
async def fine_tune_model(train_data, source_lat, source_long):
    try:
        # Fine-tune on both training data and source coordinates
        combined_prompt = f"Fine-tune the model with the following examples:\n\n{train_data}\n\nSource coordinates: {source_lat}, {source_long}"
        response = await openai.Completion.create(
            model="text-davinci-003",
            prompt=combined_prompt,
            temperature=0.7,
            max_tokens=2048,
            n=5,  # Number of completions for fine-tuning
            stop="\n"
        )
        return response
    except Exception as e:
        print(f"Fine-tuning error: {e}")
        return None
    
# Function to parse directions from HTML response
def parse_directions(html_content):
    # Implement logic to parse directions from HTML content
    # This could involve using BeautifulSoup or similar libraries
    # For demonstration purposes, we'll return a placeholder value
    return "Turn left, then turn right"

# Generate directions, time, and related information
async def generate_response(prompt, model_id, source_lat, source_long, dest_lat, dest_long):
    response = await openai.Completion.create(
        model=model_id,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )
    response_text = response.choices[0].text.strip()

    # Fetch directions and map URLs from Google Maps API
    directions_html, map_html = await fetch_api_data(source_lat, source_long, dest_lat, dest_long)
    if directions_html and map_html:
        return response_text, directions_html, map_html
    else:
        return response_text, None, None

async def main():
    while True:
        source_lat = input("Enter the source latitude: ")
        source_long = input("Enter the source longitude: ")
        dest_lat = input("Enter the destination latitude: ")
        dest_long = input("Enter the destination longitude: ")
        if source_lat.lower() == "exit" or source_long.lower() == "exit" or dest_lat.lower() == "exit" or dest_long.lower() == "exit":
            break
        train_data = generate_training_data(source_lat, source_long, dest_lat, dest_long)
        # Fine-tune model for each user query (using live API data with caching)
        model_id = await fine_tune_model(train_data ,source_lat, source_long)
        if not model_id:
            print("Failed to fine-tune model. Trying again...")
            continue  # Retry on failure

        # Prompt user for input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Generate response, directions, and map URLs
        response_text, directions_html, map_html = await generate_response(user_input, model_id, source_lat, source_long, dest_lat, dest_long)
        print("Bot:", response_text)
        if directions_html:
            print("Directions HTML:", directions_html)
        if map_html:
            print("Map HTML:", map_html)

# Run the main loop asynchronously
asyncio.run(main())

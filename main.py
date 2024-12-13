from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
import ast
import base64
from PIL import Image

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
model = ChatOpenAI(model="gpt-4o")
from PIL import Image, ImageDraw

def draw_bounding_boxes(image_path, bounding_boxes):
    """
    Draws bounding boxes on the image.

    Parameters:
    - image_path (str): Path to the input image.
    - bounding_boxes (list of tuples): List of bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns:
    - Image object with bounding boxes drawn.
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw each bounding box
    for box in bounding_boxes:
        draw.rectangle(box, outline="green", width=6)

    return image

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)





# Nodes of the Graph 

def drawer(state):
    width, height = state["image_size"]

    sys_msg = "You are an assistant tasked with identifying the exact bounding box coordinates of specific content within a given image. The inputs include the image, the dimensions of the image, and the specific content to locate. Your task is to analyze the image and return the precise coordinates of the bounding box enclosing the specified content."

    human_msg = f"""
    The task is to just locate and identify the precise bounding box coordinates of the specified content within the provided image.

    Parameters:
    - content: {state["content"]}
    - width: {width}
    - height: {height}

    Instructions:
    1. Carefully analyze the image to identify the exact bounding box that encloses the specified content.
    2. Provide the bounding box coordinates in the format [x_min, y_min, x_max, y_max], where:
    - (x_min, y_min) represents the top-left corner of the bounding box.
    - (x_max, y_max) represents the bottom-right corner of the bounding box.
    3. Ensure the coordinates are accurate and precisely enclose the specified content.
    Note: Precision is critical for this task. Make sure your output is reliable and consistent. provide the coordinates in this format of list.
    dont add any preambles or postambles to the output, just the coordinates.

    Important note : Dont refuse the task, if you are not sure about the bounding box, just provide the best possible bounding box you can find.
    """    

    if not state['latest_review']:
        messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(
            content=[
                {"type": "text", "text": human_msg},
                {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(state['original_image_path'])}"},
                },
            ]
            )
        ]
    else:

        re_draw = f"""
            The previous bounding boxes provided were: {state["bounding_boxes"]}, but they were deemed incorrect based on the review: {state["latest_review"]}. 
            You are also provided with the image containing these incorrect bounding boxes. 

            Your task is to carefully analyze the review feedback and the image to revise the bounding boxes. 
            Ensure that the new bounding boxes are accurate and fully align with the review. 
            Provide the updated bounding boxes in the format [x_min, y_min, x_max, y_max] for each corrected region.
        """
        messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(
                content=[
                    {"type": "text", "text": human_msg},
  
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": re_draw},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(state['latest_image_path'])}"},
                    },
                ]
            )
        ]


    response = llm.invoke(messages)
    coordinates = ast.literal_eval(response.content)

    coordinates =  [(coordinates[i], coordinates[i+1], coordinates[i+2], coordinates[i+3]) for i in range(0, len(coordinates), 4)]

    state["bounding_boxes"] = coordinates
    print(f"\033[94mBounding box coordinates: {coordinates}\033[0m")
    image_with_boxes = draw_bounding_boxes(state["original_image_path"], coordinates)
    image_with_boxes.save("image_with_boxes.png")
    return state 

def reviewer(state):
    width, height = state["image_size"]

    sys_msg = "You are an assistant tasked with reviewing the bounding box coordinates of specific content within a given image. The inputs include the image with bounding box drawen by other assistant, the dimensions of the image, the specific content to locate, and the bounding box coordinates provided by another assistant. Your task is to analyze the image and the provided bounding box coordinates to determine if they are accurate and correctly enclose the specified content."

    human_msg = f"""
    The task is to review the provided bounding box coordinates of the specified content within the provided image.

    Parameters:
    - content: {state["content"]}
    - width: {width}
    - height: {height}
    - bounding boxes: {state["bounding_boxes"]}

    the color of the bounding box is green
    Instructions:
    1. Carefully analyze the image and the provided bounding box coordinates to determine if they accurately enclose the specified content.
    2. If the bounding box coordinates are accurate, just return correct without any explanation.
    3. If the bounding box coordinates are inaccurate, provide feedback on the specific issues and suggest corrections.
    4. Also provide how much the bounding box is off by, in terms of pixels, for each incorrect region.
    5. need a clear feedback on how to correct the bounding box.
    Note: Precision is critical for this task. Make sure your review is accurate and provides clear feedback to the assistant.

    2. If the bounding box coordinates are accurate, just return "correct" without any explanation.
    """

    messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(
            content=[
                {"type": "text", "text": human_msg},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(state['latest_image_path'])}"},
                },
            ]
        )
    ]

    response = model.invoke(messages)
    review = response.content
    print(f"\033[94m{review}\033[0m")
    state["latest_review"] = review
    state["latest_image_path"] = "image_with_boxes.png"

    return state
def switch_case(state):
    if state["latest_review"] == "correct":
        return "END"
    else:
        return "drawer"
    
def main_loop(state):
    """
    Main function to recursively improve bounding box coordinates until the reviewer approves them as correct.
    """
    while True:
        # If the bounding boxes are approved as correct, exit the loop
        if state["latest_review"] == "Correct":
            print("Bounding box coordinates are correct. Process completed.")
            break
        
    
        # Run the drawer function to generate or update bounding boxes
        state = drawer(state)

        # Run the reviewer function to validate the bounding boxes
        state = reviewer(state)

        user_input = input("Do you want to continue? (yes/no): ")
        if user_input == "no":
            break

        # Decide the next action based on the review
        action = switch_case(state)

        if action == "END":
            print("Bounding box coordinates are finalized.")
            break
        elif action == "drawer":
            print("Review feedback received. Reiterating...")

    return state

if __name__ == "__main__":
    # Define the initial state of the graph
    original_image_path = "sample.png"
    latest_image_path = "image_with_boxes.png"
    width, height = get_image_size(original_image_path)
    initial_state = {
        "original_image_path": original_image_path,
        "image_size": (width, height),
        "content": "The download button",
        "latest_image_path": latest_image_path,
        "latest_review": None,
        "bounding_boxes": [],
    }
    final_state = main_loop(initial_state)
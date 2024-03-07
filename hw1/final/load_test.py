import numpy as np
from PIL import Image  # for image processing
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import open3d as o3d
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "/home/widden/NCTU/Perception_and_Decision_Making/dataset/apartment_0/habitat/mesh_semantic.ply"

# initialize library
sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    # "sensor_pitch": -(45 * np.pi/180)
}


def transform_rgb_bgr(image):  # chane img order from rgb(pillow) to bgr(for opencv)
    return image[:, :, [2, 1, 0]]


def transform_depth(image):  # chane origin depth infromation to img we want
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):  # 語意分割初始化(沒看很細)
    # create new img with given shape and mode(mode p)
    semantic_img = Image.new(
        "P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

# This function generates a config for the simulator.(configuration 儲存使用者的一些設定)
# It contains two parts:
# one for the simulator backend, one for the agent, where you can attach a bunch of sensors


def make_simple_cfg(settings):  # input sim_setting
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    # name is set by above library (test_scene)
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # BEV_RGB
    BEV_rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    BEV_rgb_sensor_spec.uuid = "BEV_color_sensor"
    BEV_rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    BEV_rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    BEV_rgb_sensor_spec.position = [0.0, settings["sensor_height"], -1.3]
    BEV_rgb_sensor_spec.orientation = [
        -(90 * np.pi/180),
        0.0,
        0.0,
    ]
    BEV_rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # depth snesor
    BEV_depth_sensor_spec = habitat_sim.CameraSensorSpec()
    BEV_depth_sensor_spec.uuid = "BEV_depth_sensor"
    BEV_depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    BEV_depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    BEV_depth_sensor_spec.position = [0.0, settings["sensor_height"], -1.3]
    BEV_depth_sensor_spec.orientation = [
        -(90 * np.pi/180),
        0.0,
        0.0,
    ]
    BEV_depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [
        rgb_sensor_spec, BEV_rgb_sensor_spec, BEV_depth_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(
    cfg.agents[sim_settings["default_agent"]].action_space.keys())  # 將動作指令存為list

print("Discrete action space: ", action_names)


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"
SAVE_FRONT_IMG = "p"
SAVE_BEV_IMAGE = "o"
SAVE_MANY_FRONT_IMG = "r"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


def navigateAndSee(action=""):
    global front_image, BEV_image, Depth_image,  posit
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("BEV_RGB", transform_rgb_bgr(
            observations["BEV_color_sensor"]))
        # cv2.imshow("BEV_depth", transform_depth(
        #     observations["BEV_depth_sensor"]))
        cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        # cv2.imshow("semantic", transform_semantic(
        #     observations["semantic_sensor"]))
        agent_state = agent.get_state()  # agent 更新位置？
        sensor_state = agent_state.sensor_states['color_sensor']
        # rw rx ry rz here are four elements
        posit = [sensor_state.position[0], sensor_state.position[1], sensor_state.position[2],
                 sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z]
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0], sensor_state.position[1], sensor_state.position[2],
              sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        x_temp, y_temp, z_temp, rw_temp, rx_temp, ry_temp, rz_temp = sensor_state.position[0], sensor_state.position[
            1], sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z
        front_image, BEV_image, Depth_image = transform_rgb_bgr(
            observations["color_sensor"]), transform_rgb_bgr(observations["BEV_color_sensor"]), transform_depth(observations["depth_sensor"])


def store_posit(posit_data, posit):
    with open("posit_data.txt", "a") as f:
        f.write(str(posit[0]) + " " + str(posit[1]) +
                " " + str(posit[2]) + " " + "\n")


action = "move_forward"
navigateAndSee(action)
path_pkg = "./images/"
count_rgb = 1
count_depth = 1
path_multi_rgb = []
path_multi_depth = []
posit_data = []
if not (os.path.exists("./images")):
    os.makedirs('images')
if not (os.path.exists("./images_for_projection")):
    os.makedirs('images_for_projection')

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):  # ord()return ASCII value
        action = "move_forward"
        navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    elif keystroke == ord(SAVE_FRONT_IMG):
        cv2.imwrite('./images_for_projection/front_img.png', front_image)
        print("action: save front image ")
        print(posit)

    elif keystroke == ord(SAVE_BEV_IMAGE):
        cv2.imwrite('./images_for_projection/BEV_img.png', BEV_image)
        cv2.imwrite('./images_for_projection/Depth_img.png', Depth_image)
        print("action: save BEV image")
        print(posit)
    elif keystroke == ord(SAVE_MANY_FRONT_IMG):
        path_multi_rgb.append(os.path.join(
            path_pkg, 'front_rgb_img' + '_' + str(count_rgb) + '.png'))
        path_multi_depth.append(os.path.join(
            path_pkg, 'front_depth_img' + '_' + str(count_depth) + '.png'))
        print(path_multi_depth)
        cv2.imwrite(path_multi_rgb[count_rgb-1], front_image)
        cv2.imwrite(path_multi_depth[count_depth-1], Depth_image)
        print("action: save {} th front_rgb_image".format(count_rgb))
        print("action: save {} th front_depth_img".format(count_depth))
        count_rgb = count_rgb + 1
        count_depth = count_depth + 1
        store_posit(posit_data, posit)
        print(posit)

    else:
        print("INVALID KEY")
        continue
print("load.py finish")
cv2.destroyAllWindows()

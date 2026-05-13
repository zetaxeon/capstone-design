 #!/usr/bin/env python

"""

CARLA 환경 래퍼.

- Town03 로드

- 단일 차량(hero) + 기본 센서 스폰

- reset() 시 동일 spawn point로 재스폰

"""


import logging

import carla



class CarlaEnv:

    def __init__(self, host="localhost", port=2000, town="Town03", dt=0.05):

        self.client = carla.Client(host, port)

        self.client.set_timeout(60.0)


        self.world = self.client.load_world(town)

        self.map = self.world.get_map()

        self.bp_lib = self.world.get_blueprint_library()


        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = dt

        self.world.apply_settings(settings)


        self.tm = self.client.get_trafficmanager(8000)

        self.tm.set_synchronous_mode(True)


        self.vehicle = None

        self.sensors = []

        self.collision_flag = False

        self.spawn_transform = None


    def spawn_vehicle(self, spawn_transform, vehicle_filter="vehicle.lincoln.mkz_2017"):

        self.spawn_transform = spawn_transform


        bp = self.bp_lib.filter(vehicle_filter)[0]

        bp.set_attribute("role_name", "hero")


        self.vehicle = self.world.try_spawn_actor(bp, spawn_transform)

        if self.vehicle is None:

            raise RuntimeError("Vehicle spawn failed (blocked spawn point?)")


        self._attach_sensors()

        logging.info("Spawned vehicle id=%d at %s", self.vehicle.id, spawn_transform.location)

        return self.vehicle


    def _attach_sensors(self):

        col_bp = self.bp_lib.find("sensor.other.collision")

        col = self.world.spawn_actor(

            col_bp, carla.Transform(), attach_to=self.vehicle

        )

        col.listen(lambda event: self._on_collision(event))

        self.sensors.append(col)


        cam_bp = self.bp_lib.find("sensor.camera.rgb")

        cam_bp.set_attribute("image_size_x", "400")

        cam_bp.set_attribute("image_size_y", "200")

        cam_bp.set_attribute("fov", "90")


        cam_tf = carla.Transform(

            carla.Location(x=-4.5, z=2.5),

            carla.Rotation(pitch=-20),

        )


        cam = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)

        self.sensors.append(cam)


    def _on_collision(self, event):

        self.collision_flag = True

        logging.warning("Collision with %s", event.other_actor.type_id)


    def tick(self):

        self.world.tick()


    def reset(self):

        if self.vehicle is None or self.spawn_transform is None:

            return


        self.vehicle.set_transform(self.spawn_transform)

        self.vehicle.set_target_velocity(carla.Vector3D())

        self.vehicle.set_target_angular_velocity(carla.Vector3D())

        self.vehicle.apply_control(carla.VehicleControl())

        self.collision_flag = False


        for _ in range(5):

            self.world.tick()


        logging.info("Reset vehicle to spawn point.")


    def destroy(self):

        for sensor in self.sensors:

            if sensor.is_alive:

                sensor.destroy()


        self.sensors.clear()


        if self.vehicle is not None and self.vehicle.is_alive:

            self.vehicle.destroy()

            self.vehicle = None


        if self.world is not None and self.original_settings is not None:

            self.world.apply_settings(self.original_settings) 
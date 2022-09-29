import math
import json
import time
import uuid
import random
import threading
import configparser

import fabric
import openstack

CONFIG_FILE = "./config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

MACHINES_FILE = json.load(open("./machines.json"))

MACHINES_PARADIGMS = MACHINES_FILE["machines"]
KEYPAIRS_PARADIGMS = MACHINES_FILE["keypairs"]
SECGROUPS_PARADIGMS = MACHINES_FILE["security_groups"]

# Initialize and turn on debug logging
openstack.enable_logging(debug=False)

# Initialize connections
conns = {}

VERBOSE = False

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

class Report():
    def __init__(self, status, data, errors, msg, it, t):
        
        uniques = list(set(errors))
        err = {}
        for e in uniques:
            err[str(e)] = errors.count(e)

        self.status = status
        self.data = data
        self.errors = err
        self.msg = msg
        self.iterations = it
        self.time = t

    def to_dict(self):

        data = self.data
        try:
            data = data.to_dict()
        except:
            pass
        return {
            "status": self.status,
            "data": data,
            "errors": self.errors,
            "msg": self.msg,
            "iterations": self.iterations,
            "time": self.time
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return self.to_json()

    def get_status(self):
        return self.status

    def get_data(self):
        return self.data

    def get_errors(self):
        return self.errors

    def get_msg(self):
        return self.msg

    def get_iterations(self):
        return self.iterations

    def get_time(self):
        return self.time

def debug_print(*args, **kwargs):
    verbose = kwargs.get("verbose", VERBOSE)
    try:
        kwargs.pop("verbose")
    except KeyError:
        pass
    if verbose:
        print("["+str(time.strftime("%Y-%m-%d %H:%M:%S"))+"]", flush=True, *args, **kwargs)

def report(status, data, errors, msg, it, t):
    return Report(status, data, errors, msg, it, t)

def connect_clouds(clouds, *, verbose=VERBOSE):
    if clouds is not None:
        if type(clouds) is str:
            clouds = [clouds]
        for cloud in clouds:
            if cloud not in conns:
                try:
                    conns[cloud] = openstack.connect(cloud=cloud)
                    debug_print(f"Connected to {cloud}", verbose=verbose)
                except Exception as e:
                    debug_print(f"Error connecting to {cloud}: {e}", verbose=verbose)

def get_clouds(*, verbose=VERBOSE):
    return conns.keys()

FLAVORS_HEURISTICS = { # Heuristics for sorting flavors, extendible by the user
    "cpu-ram-disk": lambda x: (x.vcpus, x.ram, x.disk), #order by smallest vcpus, then smallest ram, then smallest disk
    "cpu-disk-ram": lambda x: (x.vcpus, x.disk, x.ram),
    "ram-cpu-disk": lambda x: (x.ram, x.vcpus, x.disk),
    "ram-disk-cpu": lambda x: (x.ram, x.disk, x.vcpus),
    "disk-cpu-ram": lambda x: (x.disk, x.vcpus, x.ram),
    "disk-ram-cpu": lambda x: (x.disk, x.ram, x.vcpus),
}

def find_flavor(cloud, vcpus, ram, disk, heuristic="cpu-ram-disk", *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    conn = conns[cloud]
    available = []
    for flavor in conn.list_flavors():
        if flavor.vcpus >= vcpus and flavor.ram >= ram and flavor.disk >= disk:
            available.append(flavor)
    if len(available) == 0:
        return None
    else:
        try:
            available.sort(key=FLAVORS_HEURISTICS[heuristic])
        except Exception as e:
            debug_print(f"Error sorting flavors: {e}", verbose=verbose)

        return available[0]

def get_geographical_coordinates(address):
    try:
        import requests
        import urllib.parse
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'

        response = requests.get(url).json()
        return float(response[0]["lat"]), float(response[0]["lon"])
    except Exception as e:
        return None,None

def geo_distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p)*math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p)) / 2
    return 12742 * math.asin(math.sqrt(hav))

def find_cloud_by_address(address, *, verbose=VERBOSE):
    lat,lon = get_geographical_coordinates(address)
    return find_cloud(lat,lon,verbose=verbose)

def find_cloud(latitude, longitude, *, verbose=VERBOSE):
    min_dist = None
    min_cloud = None
    if "clouds" in MACHINES_FILE:
        for c,data in MACHINES_FILE["clouds"].items():
            #find nearest cloud
            if "latitude" not in data or "longitude" not in data:
                if "address" in data:
                    lat, lon = get_geographical_coordinates(data["address"])
                    if lat is None or lon is None:
                        continue
                    if "latitude" not in data:
                        data["latitude"] = lat
                    if "longitude" not in data:
                        data["longitude"] = lon
                else:
                    continue
            if "latitude" in data and "longitude" in data:
                dist = geo_distance(latitude, longitude, data["latitude"], data["longitude"])
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_cloud = c
    return min_cloud

def get_servers(by_metadata={}, clouds=None, tokens={}, *, verbose=VERBOSE):
    connect_clouds(clouds, verbose=verbose)
    clouds = get_clouds(verbose=verbose) if clouds is None else ([clouds] if type(clouds) is str else clouds)

    new_metadata = {}
    for k, v in by_metadata.items():
        new_metadata[apply_tokens(k, tokens)] = apply_tokens(v, tokens)
    by_metadata = new_metadata

    servers = []

    if len(by_metadata.keys()) == 0:
        for cloud in clouds:
            for server in conns[cloud].compute.servers():
                servers.append(server)
    
    else:
        for cloud in clouds:
            for server in conns[cloud].compute.servers():
                flag = True
                for key, value in by_metadata.items():
                    if key in server.metadata:
                        if type(value) is list:
                            if server.metadata[key] not in value:
                                flag = False
                        elif server.metadata[key] != value:
                            flag = False
                    else:
                        flag = False
                if flag:
                    servers.append(server)
    
    return servers

def get_server(server_name, cloud=None, *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    if cloud is None or cloud in get_clouds(verbose=verbose):
        clouds = get_clouds(verbose=verbose) if cloud is None else [cloud]
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                if server.name == server_name:
                    return server
        debug_print(f"Server {server_name} not found", verbose=verbose)
        return None
    else:
        debug_print(f"Cloud {cloud} not found", verbose=verbose)
        return None

def get_server_cloud(server_name, *, verbose=VERBOSE):
    for cloud in get_clouds(verbose=verbose):
        if get_server(server_name, cloud, verbose=verbose):
            return cloud
    debug_print(f"Server {server_name} not found in any cloud", verbose=verbose)
    return None

def get_server_type(server_name, cloud=None, *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    server = get_server(server_name, cloud, verbose=verbose)
    if server:
        try:
            return server.metadata["type"]
        except Exception as e:
            debug_print(f"Error getting server type {server_name}: {e}", verbose=verbose)
            return None
    else:
        return None

def get_server_flavor(server_name, cloud=None, *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    server = get_server(server_name, cloud, verbose=verbose)
    if server:
        return server.flavor.name
    else:
        return None

def get_server_metadata(server_name, cloud=None, *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    server = get_server(server_name, cloud, verbose=verbose)
    if server:
        return server.metadata
    else:
        return {}

def get_ip(server, *, verbose=VERBOSE):
    if server:
        try:
            for data in server.addresses["default"]:
                if data["addr"] != "" and (not data["addr"].startswith("192.168")) and (not data["addr"].startswith("10.")):
                    return data["addr"]
        except Exception as e:
            debug_print(f"Error getting server ip {server.name}: {e}", verbose=verbose)
            return None
    else:
        return None

def get_server_ip(server_name, cloud=None, *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    server = get_server(server_name, cloud, verbose=verbose)
    return get_ip(server)

def delete_server(conn, server, sleep=int(config["TUNING"]["wait_after_delete"]), max_it=int(config["TUNING"]["max_it_vm_delete"]), *, verbose=VERBOSE):
    errors = []
    for i in range(max_it):
        try:
            start = time.time()
            server.delete(session=conn.compute)
            end = time.time()
            debug_print(f"Server {server.name} deleted", verbose=verbose)
            time.sleep(sleep)
            return report(True, server.to_dict(), errors, f"Server {server.name} deleted", i+1, end-start)
        except Exception as e:
            debug_print(f"Error deleting server {server.name}: {e}", verbose=verbose)
            errors.append(str(e))
            time.sleep(sleep)
    
    return report(False, str(server), errors, f"Error deleting server {server.name}", i+1, 0)

def remove_machine(server_name, cloud=None, sleep=int(config["TUNING"]["wait_after_delete"]), max_it=int(config["TUNING"]["max_it_vm_delete"]), *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    if cloud is None:
        cloud = get_server_cloud(server_name, verbose=verbose)
        if cloud is None:
            return report(False, None, [], f"Server {server_name} cloud not found", 0, 0)
    if cloud in get_clouds(verbose=verbose):
        server = conns[cloud].compute.find_server(server_name)
        if server:
            return delete_server(conns[cloud], server, sleep, max_it, verbose=verbose)
        else:
            debug_print(f"Server {server_name} not found in {cloud}", verbose=verbose)
            return Report(False, None, [], f"Server {server_name} not found in {cloud}", 0, 0)
    else:
        debug_print(f"Cloud {cloud} not found", verbose=verbose)
        return Report(False, None, [], f"Cloud {cloud} not found", 0, 0)

def remove_all_machines(by_metadata={}, clouds=None, tokens={}, sleep=int(config["TUNING"]["wait_after_delete"]), max_it=int(config["TUNING"]["max_it_vm_delete"]), *, verbose=VERBOSE, sure = False):
    connect_clouds(clouds, verbose=verbose)
    clouds = get_clouds(verbose=verbose) if clouds is None else ([clouds] if type(clouds) is str else clouds)
    threads = []
    results = []

    new_metadata = {}
    for k, v in by_metadata.items():
        new_metadata[apply_tokens(k, tokens)] = apply_tokens(v, tokens)
    by_metadata = new_metadata

    if len(by_metadata.keys()) == 0 and sure:
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                t = ThreadWithReturnValue(target=remove_machine, args=(server.name, cloud, sleep, max_it), kwargs={"verbose":verbose})
                t.start()
                threads.append(t)

    elif len(by_metadata.keys()) > 0:
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                flag = True
                for key, value in by_metadata.items():
                    if key in server.metadata:
                        if type(value) is list:
                            if server.metadata[key] not in value:
                                flag = False
                        elif server.metadata[key] != value:
                            flag = False
                    else:
                        flag = False
                if flag:
                    t = ThreadWithReturnValue(target=remove_machine, args=(server.name, cloud, sleep, max_it), kwargs={"verbose":verbose})
                    t.start()
                    threads.append(t)
    else:
        debug_print(f"No machine removed, please set the 'sure' flag to remove all the machines", verbose=verbose)
        return results

    for t in threads:
        results.append(t.join())
    
    return results

def create_server(name, type, cloud, metadata={}, heuristic="cpu-ram-disk", max_attempts=int(config["TUNING"]["max_it_vm_create"]), timeout=float(config["TUNING"]["timeout_vm_create"]), sleep_time=int(config["TUNING"]["wait_vm_create"]), *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    errors = []
    it = 0
    if cloud is not None and cloud in get_clouds(verbose=verbose):
        conn = conns[cloud]
        s = get_server(name, cloud, verbose=False)
        print("Checking if server already exists", name, s)
        if s is not None:
            debug_print(f"Attention: Server {name} already exists in {cloud}", verbose=verbose)
            print("Deleted",delete_server(conn, s, verbose=verbose))
            time.sleep(sleep_time)
        vm = None
        if "metadata" in MACHINES_PARADIGMS[type]:
            meta = MACHINES_PARADIGMS[type]["metadata"]
        else:
            meta = {}

        for k,v in metadata.items(): # merge metadata with priority to metadata argument than with default metadata in MACHINES_PARADIGMS
            meta[k] = v

        meta["type"] = type
        meta["cloud"] = cloud
        meta["name"] = name

        if "flavor" in MACHINES_PARADIGMS[type]:
            flavor = MACHINES_PARADIGMS[type]["flavor"]
        else:
            specs = MACHINES_PARADIGMS[type]["specs"]
            flavor = find_flavor(cloud, specs["vcpus"], specs["ram"], specs["disk"], heuristic=heuristic, verbose=verbose)
            if flavor is None:
                return report(status=False, data=None, errors=errors, msg=f"Could not find flavor for {type} ({specs['vcpus']} vcpus, {specs['ram']} ram, {specs['disk']} disk)", it=0, t=0)
            else:
                MACHINES_PARADIGMS[type]["flavor"] = flavor.name
                debug_print(f"Found flavor {flavor.name} ({flavor.vcpus} vcpus, {flavor.ram} ram, {flavor.disk} disk) for {type} ({specs['vcpus']} vcpus, {specs['ram']} ram, {specs['disk']} disk)", verbose=verbose)
                flavor = flavor.name

        if "network" not in MACHINES_PARADIGMS[type]:
            MACHINES_PARADIGMS[type]["network"] = "default"

        if "security_groups" not in MACHINES_PARADIGMS[type]:
            MACHINES_PARADIGMS[type]["security_groups"] = ["default"]

        while vm is None and it < max_attempts:
            try:
                s = get_server(name, cloud, verbose=False)
                print("1Checking if server already exists", name, s)
                if s is not None:
                    debug_print(f"Attention: Server {name} already exists in {cloud}", verbose=verbose)
                    print("Deleted",delete_server(conn, s, verbose=verbose))
                    time.sleep(sleep_time)
                start = time.time()
                vm = conn.create_server(name=name, 
                                image=MACHINES_PARADIGMS[type]["image"], 
                                flavor=flavor, 
                                key_name=MACHINES_PARADIGMS[type]["key_name"], 
                                security_groups=MACHINES_PARADIGMS[type]["security_groups"],
                                network=MACHINES_PARADIGMS[type]["network"],
                                meta=meta,
                                wait=True, 
                                auto_ip=True,
                                timeout=timeout
                                )
                end = time.time()
                debug_print(f"Server {name} created", verbose=verbose)
                return report(status=True, data=vm.to_dict(), errors=errors, msg=f"Server {name} created", it=it+1, t=end-start)
            except Exception as e:
                debug_print(f"Error creating server {name}: {e}", verbose=verbose)
                it += 1
                errors.append(str(e))
                time.sleep(sleep_time)
                remove_machine(name, cloud, verbose=verbose)
        err_msg = "Error creating server {name}".format(name=name)
    else:
        debug_print(f"Cloud {cloud} not found", verbose=verbose)
        err_msg = "Cloud {cloud} not found".format(cloud=cloud)
    return report(status=False, data=None, errors=errors, msg=err_msg, it=it+1, t=0)

# Python3 program to find the smallest window
# containing all characters of a pattern.
no_of_chars = 256
 
# Function to find smallest window
# containing all characters of 'pat'
def findSubString(string, pat, starts_with="", ends_with=""):
 
    pat = starts_with + pat + ends_with

    len1 = len(string)
    len2 = len(pat)
 
    # Check if string's length is
    # less than pattern's
    # length. If yes then no such
    # window can exist
    if len1 < len2:
        return None
 
    hash_pat = [0] * no_of_chars
    hash_str = [0] * no_of_chars
 
    # Store occurrence ofs characters of pattern
    for i in range(0, len2):
        hash_pat[ord(pat[i])] += 1
 
    start, start_index, min_len = 0, -1, float('inf')
 
    # Start traversing the string
    count = 0  # count of characters
    for j in range(0, len1):
 
        # count occurrence of characters of string
        hash_str[ord(string[j])] += 1
 
        # If string's char matches with
        # pattern's char then increment count
        if (hash_str[ord(string[j])] <=
                hash_pat[ord(string[j])]):
            count += 1
 
        # if all the characters are matched
        if count == len2:
 
            # Try to minimize the window
            while (hash_str[ord(string[start])] >
                   hash_pat[ord(string[start])] or
                   hash_pat[ord(string[start])] == 0):
 
                if (hash_str[ord(string[start])] >
                        hash_pat[ord(string[start])]):
                    hash_str[ord(string[start])] -= 1
                start += 1
 
            # update window size
            len_window = j - start + 1

            if min_len > len_window  and (starts_with == "" or string[start] == starts_with) and (ends_with == "" or min_len==float('inf') or string[start+min_len] == ends_with):
 
                min_len = len_window
                start_index = start
 
    # If no window found
    if start_index == -1:
        return None
 
    # Return substring starting from
    # start_index and length min_len
    return string[start_index: start_index + min_len]
 

SEMANTIC_TOKENS_STATUS = {
}

def semantic_token_inc(token, server_name, cloud):
    original = ""
    for t in token:
        original += str(t)+"_"
    original = original[:-1]
    if original not in SEMANTIC_TOKENS_STATUS:
        SEMANTIC_TOKENS_STATUS[original] = 0
    else:
        SEMANTIC_TOKENS_STATUS[original] += 1
    return SEMANTIC_TOKENS_STATUS[original]

def semantic_token_id(token, server_name, cloud):
    original = ""
    for t in token:
        original += str(t)+"_"
    original = original[:-1]
    if original not in SEMANTIC_TOKENS_STATUS:
        SEMANTIC_TOKENS_STATUS[original] = uuid.uuid4()
    return SEMANTIC_TOKENS_STATUS[original]

SEMANTIC_TOKENS = { #extendible by the user
    "CLOUD": {
        "elements": 0,
        "function": lambda token,server_name,cloud: get_server_cloud(server_name) if server_name is not None else cloud
    },
    "HOSTNAME": {
        "elements": 0,
        "function": lambda token,server_name,cloud: server_name
    },
    "ID": {
        "elements": 1,
        "function": semantic_token_id
    },
    "RANDOM": {
        "elements": 2,
        "function": lambda token,server_name,cloud: random.randint(int(token[1]), int(token[2]))
    },
    "INC": {
        "elements": 1,
        "function": semantic_token_inc
    }
}

def apply_semantic_tokens(string, server_name=None, cloud=None):
    global SEMANTIC_TOKENS_STATUS
    for pat in SEMANTIC_TOKENS.keys():
        size = int((SEMANTIC_TOKENS[pat]["elements"]))
        exist = True
        while exist:
            token = findSubString(string, pat+("_"*size), starts_with="<", ends_with=">")
            if token is not None:
                original = token
                token = token.replace("<", "").replace(">", "").split("_")

                try:
                    if len(token) == size+1:
                        value = SEMANTIC_TOKENS[pat]["function"](token,server_name,cloud)
                        if value is not None:
                            string = string.replace(original, str(value), 1)
                        else:
                            exist = False
                except Exception as e:
                    debug_print(f"Error applying semantic token {pat} to string {string}, server_name={server_name}, cloud={cloud}: {e}")
                    exist = False
            else:
                exist = False
    return string

def apply_tokens(string, tokens, server_name=None, cloud=None):
    if type(string) is str:
        string = apply_semantic_tokens(string, server_name=server_name, cloud=cloud)
        for k,v in tokens.items():
            string = string.replace(k, v)
    elif type(string) is list:
        for i in range(len(string)):
            string[i] = apply_tokens(string[i], tokens, server_name=server_name, cloud=cloud)
    return string

def apply_tokens_list(script, tokens, server_name=None, cloud=None):
    script = script.copy()
    for i in range(len(script)):
        script[i] = apply_tokens(script[i], tokens, server_name=server_name, cloud=cloud)
    return script
    
def exec_script_by_ip(ip, script, user, key_filename, tokens={}, hide=True, timeout_cmd=int(config["TUNING"]["script_connect_timeout"]), max_script_attempts=int(config["TUNING"]["max_it_vm_execute_script"]), sleep_script=int(config["TUNING"]["wait_vm_execute_script"]), max_cmd_attempts=int(config["TUNING"]["max_it_vm_execute_cmd"]), sleep_cmd=int(config["TUNING"]["wait_vm_execute_cmd"]), server_name = "", *, verbose=VERBOSE):
    script = apply_tokens_list(script, tokens, server_name=server_name)
    debug_print(f"Executing script on {server_name} {ip}: {script}", verbose=verbose)
    retry = True
    errors = []
    it = 0
    msg = ""
    last = None    
    if type(script) is not list:
        debug_print(f"Script {script} for {server_name} is not a list", verbose=verbose)
        return report(status=False, data=None, errors=errors, msg=f"Script {script} for {server_name} is not a list", it=0, t=0)

    while retry:
        debug_print("Trying to connect to {} ({})".format(server_name, ip), verbose=verbose)
        start = time.time()
        try:
            if timeout_cmd != 0:
                vm = fabric.Connection(ip, 
                                    user=user,
                                    connect_kwargs={
                                            "key_filename": key_filename,
                                            },
                                    connect_timeout = timeout_cmd,
                                    )
            else:
                vm = fabric.Connection(ip, 
                                    user=user,
                                    connect_kwargs={
                                            "key_filename": key_filename,
                                            },
                                    )
            debug_print("Connected to {} ({})".format(server_name, ip), verbose=verbose)
            for cmd in script:
                time.sleep(sleep_cmd)
                debug_print("Executing command {} on {} ({})".format(cmd, server_name, ip), verbose=verbose)
                t = 0
                exc = None
                while t < max_cmd_attempts:
                    try:
                        if timeout_cmd != 0:
                            last = vm.run(cmd, hide=hide, warn=True, timeout=timeout_cmd)
                        else:
                            last = vm.run(cmd, hide=hide, warn=True)
                        break
                    except Exception as e1:
                        debug_print("Error while '{}': {} on {} ({}) [{}/{}]".format(cmd, e1, server_name, ip, t+1, max_cmd_attempts), verbose=verbose)
                        errors.append(str(e1))
                        t += 1
                        exc = e1
                        time.sleep(1)
                if exc is not None and t == max_cmd_attempts:
                    debug_print("Failed to connect to {} ({})".format(server_name, ip), verbose=verbose)
                    raise exc
            end = time.time()
            debug_print("Script executed on {} ({})".format(server_name, ip), verbose=verbose)
            return report(status=True, data=str(last), errors=errors, msg=f"Script executed on {server_name} ({ip})", it=it+1, t=end-start) #TODO: last to dict?
        except Exception as e:
            debug_print(f"Error connecting to {server_name} ({ip}): {e}", verbose=verbose)
            errors.append(str(e))
            if len(e.args) == 2 and (not e.args[1].startswith("Unable to connect to port 22 on 90.147")):
                print("Args",e.args)
            it += 1
            msg = f"Error connecting to {server_name} ({ip}): {e}"
            if len(e.args) == 2 and any(err for err in ["Unable to connect", "Connection refused", "Encountered a bad command exit code"] if err.lower() in e.args[1].lower()): # "Timeout waiting for the server to come up"
                if it < max_script_attempts:
                    time.sleep(sleep_script)
                else:
                    retry = False
            else:
                retry = False
    return report(status=False, data=None, errors=errors, msg=msg, it=it+1, t=0)

def exec_script(server_name, script, tokens={}, hide=True, timeout_cmd=int(config["TUNING"]["script_connect_timeout"]), max_script_attempts=int(config["TUNING"]["max_it_vm_execute_script"]), sleep_script=int(config["TUNING"]["wait_vm_execute_script"]), max_cmd_attempts=int(config["TUNING"]["max_it_vm_execute_cmd"]), sleep_cmd=int(config["TUNING"]["wait_vm_execute_cmd"]), *, verbose=VERBOSE):
    ip = get_server_ip(server_name, verbose=verbose)
    if ip is None:
        debug_print(f"Server {server_name} IP not found", verbose=verbose)
        return report(status=False, data=None, errors=[], msg=f"Server {server_name} IP not found", it=0, t=0)
    debug_print("Executing script on {} ({})".format(server_name, ip), verbose=verbose)
    server_type = get_server_type(server_name)
    if server_type is None:
        debug_print(f"Server {server_name} type not found", verbose=verbose)
        return report(status=False, data=None, errors=[], msg=f"Server {server_name} type not found", it=0, t=0)
    
    try:
        key_filename = KEYPAIRS_PARADIGMS[MACHINES_PARADIGMS[server_type]["key_name"]]["key_filename"]
        user = KEYPAIRS_PARADIGMS[MACHINES_PARADIGMS[server_type]["key_name"]]["user"]
    except Exception as e:
        debug_print(f"Error getting key_filename for {server_name}: {e}", verbose=verbose)
        return report(status=False, data=None, errors=[str(e)], msg=f"Error getting key_filename for {server_name}", it=0, t=0)

    if type(script) is str:
        try:
            script = MACHINES_PARADIGMS[server_type]["scripts"][script]
        except Exception as e:
            debug_print(f"Error getting script {script} for {server_name}: {e}", verbose=verbose)
            return report(status=False, data=None, errors=[str(e)], msg=f"Error getting script {script} for {server_name} ({server_type})", it=0, t=0)

    new_tokens = {}

    try:
        new_tokens = MACHINES_FILE["tokens"]
    except Exception as e:
        pass
    try:
        for k, v in MACHINES_PARADIGMS[server_type]["tokens"].items():
            new_tokens[k] = v
    except Exception as e:
        pass

    for k, v in tokens.items():
        new_tokens[k] = v

    return exec_script_by_ip(ip, script, user, key_filename, new_tokens, hide, timeout_cmd, max_script_attempts, sleep_script, max_cmd_attempts, sleep_cmd, server_name=server_name, verbose=verbose)

def exec_script_on_thread(server_name, script, tokens={}, hide=True, timeout_cmd=int(config["TUNING"]["script_connect_timeout"]), max_script_attempts=int(config["TUNING"]["max_it_vm_execute_script"]), sleep_script=int(config["TUNING"]["wait_vm_execute_script"]), max_cmd_attempts=int(config["TUNING"]["max_it_vm_execute_cmd"]), sleep_cmd=int(config["TUNING"]["wait_vm_execute_cmd"]), *, verbose=VERBOSE):
    t = ThreadWithReturnValue(target=exec_script, args=(server_name, script, tokens, hide, timeout_cmd, max_script_attempts, sleep_script, max_cmd_attempts, sleep_cmd), kwargs={"verbose":verbose})
    t.start()
    return t  

SERVERS_HEURISTICS = {
    "enum": lambda x: x,
    "random": lambda x: random.shuffle(x),
    "alphabetical": lambda x: sorted(x, key=lambda x: x.name),
    "alphabetical_reverse": lambda x: sorted(x, key=lambda x: x.name, reverse=True)
}

def exec_script_machines(script, by_metadata={}, clouds=None, tokens={}, count=float("inf"), heuristic="enum", hide=True, timeout_cmd=int(config["TUNING"]["script_connect_timeout"]), max_script_attempts=int(config["TUNING"]["max_it_vm_execute_script"]), sleep_script=int(config["TUNING"]["wait_vm_execute_script"]), max_cmd_attempts=int(config["TUNING"]["max_it_vm_execute_cmd"]), sleep_cmd=int(config["TUNING"]["wait_vm_execute_cmd"]), *, verbose=VERBOSE):
    connect_clouds(clouds, verbose=verbose)

    new_metadata = {}
    for k, v in by_metadata.items():
        new_metadata[apply_tokens(k, tokens)] = apply_tokens(v, tokens)
    by_metadata = new_metadata

    servers = []

    threads = []
    clouds = get_clouds(verbose=verbose) if clouds is None else ([clouds] if type(clouds) is str else clouds)
    if len(by_metadata.keys()) == 0:
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                if server.status == "ACTIVE":
                    servers.append(server)
    else:
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                flag = True
                for key, value in by_metadata.items():
                    if key in server.metadata:
                        if type(value) is list:
                            if server.metadata[key] not in value:
                                flag = False
                        elif server.metadata[key] != value:
                            flag = False     
                    else:
                        flag = False
                if flag and server.status == "ACTIVE":
                    servers.append(server)

    try:
        servers = SERVERS_HEURISTICS[heuristic](servers)
    except Exception as e:
        debug_print(f"Error executing heuristic {heuristic}: {e}", verbose=verbose)

    c = 0
    for server in servers:
        if c < count:
            threads.append(exec_script_on_thread(server.name, script, tokens, hide, timeout_cmd, max_script_attempts, sleep_script, max_cmd_attempts, sleep_cmd, verbose=verbose))
            c += 1
        else:
            break

    results = []

    for t in threads:
        results.append(t.join())
    return results

def get_tasks(server_name=None, type=None):
    if type is None:
        try:
            metadata = get_server_metadata(server_name)
            type = metadata["type"]
        except Exception as e:
            debug_print(f"Error in get_tasks getting type for {server_name}: {e}")
            return None
    try:
        return MACHINES_PARADIGMS[type]["tasks"]
    except Exception as e:
        debug_print(f"Error in get_tasks getting tasks for {server_name}: {e}")
        return []

def get_task(server_name=None, type=None, name=None):
    tasks = get_tasks(server_name, type)
    if tasks is None:
        return None

    for step in tasks:
        for task in step:
            if task["name"] == name:
                return task
    return None


def add_machine(name, type, cloud, metadata={}, tokens={}, heuristic="cpu-ram-disk", random_sleep=int(config["TUNING"]["random_sleep_vm_create"]), setup_timeout=float(config["TUNING"]["timeout_vm_setup"]), max_attempts=int(config["TUNING"]["max_it_vm_setup"]), sleep=int(config["TUNING"]["sleep_vm_setup"]), max_create_attempts=int(config["TUNING"]["max_it_vm_create"]), create_timeout=float(config["TUNING"]["timeout_vm_create"]), create_sleep_time=int(config["TUNING"]["wait_vm_create"]), *, verbose=VERBOSE, verbose_script=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    time.sleep(random.random() * random_sleep)
    retry = True
    it = 0

    name = apply_tokens(name, tokens, cloud=cloud)
    new_metadata = {}
    for k, v in metadata.items():
        new_metadata[apply_tokens(k, tokens, name, cloud)] = apply_tokens(v, tokens, name, cloud)
    metadata = new_metadata

    while retry:
        result = {}
        debug_print("Creating VM {} ({}) on {}".format(name, type, cloud), verbose=verbose)
        vm = create_server(name, type, cloud, metadata, heuristic, max_create_attempts, create_timeout, create_sleep_time, verbose=verbose).to_dict()
        if vm["status"] is False:
            debug_print(f"Error creating VM {name} ({type}) on {cloud}", verbose=verbose)
            return report(status=False, data={"create":vm,"setup":None}, errors=[], msg=f"Error creating VM {name} ({type}) on {cloud}", it=it+1, t=0)
        else:
            ip = None
            start = time.time()
            while ip is None and time.time() - start < setup_timeout:
                ip = get_server_ip(name, verbose=verbose)
                time.sleep(1)
            if ip is None:
                debug_print("Error getting IP for {}".format(name), verbose=verbose)
                return report(status=False, data={"create":vm,"setup":None}, errors=[], msg=f"Timeout waiting for {name} IP", it=it+1, t=0)
            
            debug_print("VM {} ({}) created".format(name, ip), verbose=verbose)

            if "setup" not in MACHINES_PARADIGMS[type]["scripts"]:
                MACHINES_PARADIGMS[type]["scripts"]["setup"] = []

            if "files" not in MACHINES_PARADIGMS[type]:
                MACHINES_PARADIGMS[type]["files"] = []

            tasks = get_tasks(name, type)
            if tasks is not None:
                for step in tasks:
                    for task in step:
                        if "setup" in task:
                            for cmd in task["setup"]:
                                if cmd not in MACHINES_PARADIGMS[type]["scripts"]["setup"]:
                                    MACHINES_PARADIGMS[type]["scripts"]["setup"].append(cmd)
                        if "files" in task:
                            for file in task["files"]:
                                if file not in MACHINES_PARADIGMS[type]["files"]:
                                    MACHINES_PARADIGMS[type]["files"].append(file)
            delta = 0
            if "setup" in MACHINES_PARADIGMS[type]["scripts"] and len(MACHINES_PARADIGMS[type]["scripts"]["setup"]) > 0:
                debug_print("Setting up VM {} ({})".format(name, ip), verbose=verbose)
                start = time.time()
                while delta < setup_timeout:
                    result = exec_script(name, "setup", tokens=tokens, hide=(not verbose_script), timeout_cmd=setup_timeout,  max_script_attempts=1, sleep_script=1, max_cmd_attempts=1, sleep_cmd=1, verbose=False).to_dict()
                    if result["status"]:
                        break
                    delta = time.time() - start
                    time.sleep(1)
                if result["status"] is False:
                    debug_print("Error setting up VM {} ({}): {}".format(name, ip, result["msg"]), verbose=verbose)
                    remove_machine(name, verbose=verbose)
                    it += 1

                    if it >= max_attempts:
                        retry = False
                        debug_print("Max attempts reached for {}".format(name), verbose=verbose)
                        remove_machine(name, verbose=verbose)
                        return report(status=False, data={"create":vm,"setup":result}, errors=[], msg=f"Max attempts reached for {name}", it=it+1, t=0)
                    else:
                        time.sleep(sleep)
                else:
                    retry = False
            else:
                retry = False

    files = {}
    if ip is not None and "files" in MACHINES_PARADIGMS[type] and len(MACHINES_PARADIGMS[type]["files"]) > 0:
        debug_print("Copying files to VM {} ({})".format(name, ip), verbose=verbose)
        try:
            files = MACHINES_PARADIGMS[type]["files"]
            key_filename = KEYPAIRS_PARADIGMS[MACHINES_PARADIGMS[type]["key_name"]]["key_filename"]
            user = KEYPAIRS_PARADIGMS[MACHINES_PARADIGMS[type]["key_name"]]["user"]
            timeout_cmd=int(config["TUNING"]["script_connect_timeout"])

            if timeout_cmd != 0:
                c = fabric.Connection(ip, 
                                    user=user,
                                    connect_kwargs={
                                            "key_filename": key_filename,
                                            },
                                    connect_timeout = timeout_cmd,
                                    )
            else:
                c = fabric.Connection(ip, 
                                    user=user,
                                    connect_kwargs={
                                            "key_filename": key_filename,
                                            },
                                    )
            for file in files:
                try:
                    source = file["source"]
                    source = apply_tokens(source, tokens, name, cloud)
                    destination = file["destination"]
                    destination = apply_tokens(destination, tokens, name, cloud)
                    c.put(source, destination)
                    file["result"] = True
                except Exception as e:
                    file["result"] = str(e)
            debug_print("Files copied to VM {} ({})".format(name, ip), verbose=verbose)
        except Exception as e:
            debug_print("Error copying files to VM {} ({}): {}".format(name, ip, e), verbose=verbose)
            files = report(status=False, data={"create":vm,"setup":result}, errors=[str(e)], msg=f"Error connecting to {name} ({ip}) to upload files", it=it+1, t=delta).to_dict()

    debug_print("VM {} ({}) setup completed".format(name, ip), verbose=verbose)
    return report(status=True, data={"create":vm,"setup":result, "files":files}, errors=[], msg=f"VM {name} ({ip}) created", it=it+1, t=delta)

class VMThread(threading.Thread):
    def __init__(self, name, type, cloud, metadata={}, tokens={}, heuristic="cpu-ram-disk", random_sleep=int(config["TUNING"]["random_sleep_vm_create"]), setup_timeout=float(config["TUNING"]["timeout_vm_setup"]), max_attempts=int(config["TUNING"]["max_it_vm_setup"]), sleep=int(config["TUNING"]["sleep_vm_setup"]), max_create_attempts=int(config["TUNING"]["max_it_vm_create"]), create_timeout=float(config["TUNING"]["timeout_vm_create"]), create_sleep_time=int(config["TUNING"]["wait_vm_create"]), *, verbose=VERBOSE, verbose_script=VERBOSE):
        threading.Thread.__init__(self, target=add_machine, args=(name, type, cloud, metadata, tokens, heuristic, random_sleep, setup_timeout, max_attempts, sleep, max_create_attempts, create_timeout, create_sleep_time), kwargs={"verbose":verbose, "verbose_script":verbose_script})
        self._return = None
        
    def run(self):
        self._return = self._target(*self._args, **self._kwargs)
            
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

def add_machines(specs, tokens={}, heuristic="cpu-ram-disk", random_sleep=int(config["TUNING"]["random_sleep_vm_create"]), setup_timeout=float(config["TUNING"]["timeout_vm_setup"]), max_attempts=int(config["TUNING"]["max_it_vm_setup"]), sleep=int(config["TUNING"]["sleep_vm_setup"]), max_create_attempts=int(config["TUNING"]["max_it_vm_create"]), create_timeout=float(config["TUNING"]["timeout_vm_create"]), create_sleep_time=int(config["TUNING"]["wait_vm_create"]), *, verbose=VERBOSE, verbose_script=VERBOSE):
    threads = []
    for name,spec in specs.items():
        type = spec["type"]
        metadata = {}
        if "metadata" in spec:
            metadata = spec["metadata"]
        for cloud,count in spec["clouds"].items():
            connect_clouds(cloud, verbose=verbose)
            for _ in range(count):
                t = VMThread(str(name), type, cloud, metadata, tokens, heuristic, random_sleep, setup_timeout, max_attempts, sleep, max_create_attempts, create_timeout, create_sleep_time, verbose=verbose, verbose_script=verbose_script)
                t.start()
                threads.append(t)
    results = []
    for t in threads:
        results.append(t.join())
    return results

def update_server(metadata, conn, server, sleep=int(config["TUNING"]["wait_after_update"]), max_it=int(config["TUNING"]["max_it_vm_update"]), *, verbose=VERBOSE):
    errors = []
    for i in range(max_it):
        try:
            old_metadata = get_server_metadata(server.name, verbose=verbose)
            to_remove = []
            for k, v in metadata.items():
                if k not in ["type", "name", "cloud"]:
                    if (v is None or v.trim() == "" or v.lower() == "none" or v.lower() == "null"):
                        to_remove.append(k)
                        if k in old_metadata:
                            del old_metadata[k]
                    else:
                        old_metadata[k] = v
            start = time.time()
            conn.delete_server_metadata(server, to_remove)
            time.sleep(sleep)
            conn.set_server_metadata(server, old_metadata)
            end = time.time()
            debug_print(f"Server {server.name} updated", verbose=verbose)
            time.sleep(sleep)
            return report(True, server.to_dict(), errors, f"Server {server.name} updated", i+1, end-start-sleep)
        except Exception as e:
            debug_print(f"Error updating server {server.name}: {e}", verbose=verbose)
            errors.append(str(e))
    
    return report(False, str(server), errors, f"Error updating server {server.name}", i+1, 0)

def update_machine(metadata, server_name, cloud=None, sleep=int(config["TUNING"]["wait_after_update"]), max_it=int(config["TUNING"]["max_it_vm_update"]), *, verbose=VERBOSE):
    connect_clouds(cloud, verbose=verbose)
    if cloud is None:
        cloud = get_server_cloud(server_name, verbose=verbose)
        if cloud is None:
            return report(False, None, [], f"Server {server.name} cloud not found", 0, 0)
    if cloud in get_clouds(verbose=verbose):
        server = conns[cloud].compute.find_server(server_name)
        if server:
            return update_server(metadata, conns[cloud], server, sleep, max_it, verbose=verbose)
        else:
            debug_print(f"Server {server_name} not found in {cloud}", verbose=verbose)
            return Report(False, None, [], f"Server {server_name} not found in {cloud}", 0, 0)
    else:
        debug_print(f"Cloud {cloud} not found", verbose=verbose)
        return Report(False, None, [], f"Cloud {cloud} not found", 0, 0)

def update_machines(metadata, by_metadata={}, clouds=None, tokens={}, sleep=int(config["TUNING"]["wait_after_update"]), max_it=int(config["TUNING"]["max_it_vm_update"]), *, verbose=VERBOSE, sure = False):
    connect_clouds(clouds, verbose=verbose)
    clouds = get_clouds(verbose=verbose) if clouds is None else ([clouds] if type(clouds) is str else clouds)
    threads = []
    results = []

    new_metadata = {}
    for k, v in metadata.items():
        new_metadata[apply_tokens(k, tokens)] = apply_tokens(v, tokens)
    metadata = new_metadata

    new_metadata = {}
    for k, v in by_metadata.items():
        new_metadata[apply_tokens(k, tokens)] = apply_tokens(v, tokens)
    by_metadata = new_metadata

    if len(by_metadata.keys()) == 0:
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                t = ThreadWithReturnValue(target=update_machine, args=(metadata, server.name, cloud, sleep, max_it), kwargs={"verbose":verbose})
                t.start()
                threads.append(t)

    elif len(by_metadata.keys()) > 0:
        for cloud in clouds:
            for server in get_servers(clouds=cloud, verbose=verbose):
                flag = True
                for key, value in by_metadata.items():
                    if key in server.metadata:
                        if type(value) is list:
                            if server.metadata[key] not in value:
                                flag = False
                        elif server.metadata[key] != value:
                            flag = False
                    else:
                        flag = False
                if flag:
                    t = ThreadWithReturnValue(target=update_machine, args=(metadata, server.name, cloud, sleep, max_it), kwargs={"verbose":verbose})
                    t.start()
                    threads.append(t)

    for t in threads:
        results.append(t.join())
    
    return results

def setup_clouds(clouds, ips=0, *, verbose=VERBOSE):
    if clouds is not None:
        if type(clouds) is str:
            clouds = [clouds]
        connect_clouds(clouds, verbose=verbose)
        for cloud in clouds:
            conn = conns[cloud]
            for sg,info in SECGROUPS_PARADIGMS.items():
                try:
                    debug_print("Creating security group {} on cloud {}".format(sg, cloud), verbose=verbose)
                    if conn.get_security_group(sg) is not None:
                        conn.delete_security_group(sg)
                    conn.create_security_group(sg, info["description"])
                    for rule in info["rules"]:
                        if "port_range_min" in rule and "port_range_max" in rule:
                            conn.create_security_group_rule(sg,
                                                            port_range_min=rule["port_range_min"],
                                                            port_range_max=rule["port_range_max"], 
                                                            protocol=rule["protocol"], 
                                                            remote_ip_prefix=rule["remote_ip_prefix"],
                                                            direction=rule["direction"], 
                                                            ethertype=rule["ethertype"])
                        else:
                            conn.create_security_group_rule(sg,
                                                            protocol=rule["protocol"], 
                                                            remote_ip_prefix=rule["remote_ip_prefix"],
                                                            direction=rule["direction"], 
                                                            ethertype=rule["ethertype"])
                except Exception as e:
                    debug_print("Error creating security group {} on cloud {}: {}".format(sg, cloud, e), verbose=verbose)

            for key,info in KEYPAIRS_PARADIGMS.items():
                debug_print("Creating keypair {} on cloud {}".format(key, cloud), verbose=verbose)
                if conn.get_keypair(key) is not None:
                    conn.delete_keypair(key)
                conn.create_keypair(key, info["public_key"])

            debug_print(f"Allocating {ips} floating IPs on cloud {cloud}", verbose=verbose)
            n = 0
            for _ in range(ips):
                try:
                    conn.create_floating_ip()
                    n+=1
                except:
                    pass
            debug_print(f"{n} floating IPs allocated on cloud {cloud}", verbose=verbose)
            if verbose:
                print()

def status_clouds(clouds, metadata={}, *, verbose=VERBOSE):
    if clouds is not None:
        connect_clouds(clouds, verbose=verbose)
        if type(clouds) is str:
            clouds = [clouds]
        for cloud in clouds:
            conn = conns[cloud]
            print(f"Cloud {cloud}")
            print("="*len(f"Cloud {cloud}"))
            print()
            print("Servers:")
            print("="*len("Servers:"))
            print("Metadata:")
            print(metadata)
            print("="*len(str(metadata)))
            print()
            for server in conn.list_servers():
                flag = True
                for key, value in metadata.items():
                    if key in server.metadata:
                        if type(value) is list:
                            if server.metadata[key] not in value:
                                flag = False
                        elif server.metadata[key] != value:
                            flag = False
                    else:
                        flag = False
                if flag:
                    print(f"{server.name} ({get_server_flavor(server.name, cloud, verbose=verbose)}) - {get_server_ip(server.name, cloud, verbose=verbose)} [{server.status}]")
                    for key,value in server.metadata.items():
                        print(f"{key}: {value}")
                    print("-"*len(f"{server.name} ({get_server_flavor(server.name, cloud, verbose=verbose)}) - {get_server_ip(server.name, cloud, verbose=verbose)} [{server.status}]"))
            print()
            print("Security groups:")
            print("="*len("Security groups:"))
            for sg in conn.list_security_groups():
                print(f"{sg.name}")
            print()
            print("Keypairs:")
            print("="*len("Keypairs:"))
            for key in conn.list_keypairs():
                print(f"{key.name} - {key.public_key}")
                print("-"*len(f"{key.name} - {key.public_key}"))
            print()
            print("Floating IPs:")
            print("="*len("Floating IPs:"))
            n_ips = 0
            n_ips_attached = 0
            for ip in conn.list_floating_ips():
                n_ips+=1
                if ip.attached:
                    n_ips_attached+=1 
            print(f"{n_ips} floating IPs allocated")
            print(f"{n_ips_attached} floating IPs attached")
            print(f"{n_ips-n_ips_attached} floating IPs available")
            print()
            print()
            print()

def parse_list_to_dict(ls):
    d = {}
    if ls is not None:
        for v in ls:
            v = v.split("=")
            if len(v) == 2:
                key = v[0]
                value = v[1]
                if key not in d:
                    d[key] = []
                d[key].append(value)
    return d

def parse_dict_to_one_to_one_dict(d):
    new_d = {}
    for key,values in d.items():
        if type(values) is list:
            values = values[0]
        new_d[key] = values
    return new_d

class PipelineTask(threading.Thread):
    def __init__(self, wait=0, period=None, group=None, target=None, name=None,
                 args=(), kwargs={}, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._wait = wait
        self._period = period
        self._return = None
    def run(self):
        if self._target is not None:
            time.sleep(self._wait)
            while True:
                self._return = self._target(*self._args,
                                                    **self._kwargs)
                if self._period is None:
                    break
                time.sleep(self._period)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

def run(pipeline, *, verbose=VERBOSE, verbose_script=VERBOSE):
    results = []
    s = 0
    for step in pipeline:
        s+=1
        print("*"*(len(f"*** Running step {s} ***")))
        print(f"*** Running step {s} ***")
        print("*"*(len(f"*** Running step {s} ***")))
        threads = []
        step_results = []
        c = 0
        for component in step:
            c+=1
            t = None
            if "period" not in component:
                component["period"] = None
            if "wait" not in component:
                component["wait"] = 0
            if "clouds" not in component:
                    component["clouds"] = None
            if "metadata" not in component:
                component["metadata"] = {}
            if "tokens" not in component:
                component["tokens"] = {}

            if component["action"] == "create":
                if "heuristic" not in component:
                    component["heuristic"] = "cpu-ram-disk"
                print(f"* Running component {s}/{c}: creating servers *")
                t = PipelineTask(wait=component["wait"], period=component["period"], target=add_machines, args=(component["specs"],component["tokens"],component["heuristic"]), kwargs={"verbose":verbose, "verbose_script":verbose_script})
            elif component["action"] == "remove":
                print(f"* Running component {s}/{c}: removing servers *")
                t = PipelineTask(wait=component["wait"], period=component["period"], target=remove_all_machines, args=(component["metadata"], component["clouds"], component["tokens"]), kwargs={"verbose":verbose})
            elif component["action"] == "execute":
                if "count" not in component:
                    component["count"] = float("inf")
                if "heuristic" not in component:
                    component["heuristic"] = "enum"
                print(f"* Running component {s}/{c}: executing script on servers *")
                t = PipelineTask(wait=component["wait"], period=component["period"], target=exec_script_machines, args=(component["script"], component["metadata"], component["clouds"], component["tokens"], component["count"], component["heuristic"]), kwargs={"verbose":verbose})
            elif component["action"] == "update":
                print(f"* Running component {s}/{c}: updating servers *")
                t = PipelineTask(wait=component["wait"], period=component["period"], target=update_machines, args=(component["specs"], component["metadata"], component["clouds"], component["tokens"]), kwargs={"verbose":verbose})
            elif component["action"] == "setup":
                print(f"* Running component {s}/{c}: setting up servers *")
                if "ips" not in component:
                    component["ips"] = 0
                t = PipelineTask(wait=component["wait"], period=component["period"], target=setup_clouds, args=(component["clouds"], component["ips"]), kwargs={"verbose":verbose})
            elif component["action"] == "status":
                print(f"* Running component {s}/{c}: getting status of clouds *")
                t = PipelineTask(wait=component["wait"], period=component["period"], target=status_clouds, args=(component["clouds"], component["metadata"]), kwargs={"verbose":verbose})
            else:
                raise Exception("Unknown action {}".format(component["action"]))

            if t is not None:
                t.start()
                if component["period"] is None:
                    threads.append(t)
    
        for t in threads:
            step_results.append(t.join())
        
        results.append(step_results)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="machines.py - A fault resistant way to create and manage virtual machines on multiple clouds using OpenStack with QoS feedback")

    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output", default=False)
    
    # subparsers for commands
    subparsers = parser.add_subparsers(help="Subcommands", dest="command")
    subparsers.required = True

    # create subparser for setup
    setup_parser = subparsers.add_parser("setup", help="Setup the clouds")
    setup_parser.add_argument("clouds", nargs="+", help="Clouds to setup")
    setup_parser.add_argument("-i", "--ips", type=int, default=0, help="Number of floating IPs to allocate")


    # create subparser for status
    status_parser = subparsers.add_parser("status", help="Status of the clouds")
    status_parser.add_argument("clouds", nargs="+", help="Clouds to status")

    status_parser.add_argument("-m", "--metadata", metavar="KEY=VALUE", nargs='+', help="Metadata of the machines to analyse", default=None)

    # create subparser for add
    add_parser = subparsers.add_parser("add", help="Create the virtual machines")
    add_parser.add_argument("name", help="Name of the virtual machine")
    add_parser.add_argument("type", help="Type of the virtual machine")
    add_parser.add_argument("clouds", nargs="+", help="Clouds to add the virtual machines to")
    add_parser.add_argument("-m", "--metadata", metavar="KEY=VALUE", nargs="+", help="Metadata to add to the virtual machine")
    add_parser.add_argument("-t", "--tokens", metavar="KEY=VALUE", nargs="+", help="Tokens to apply to the setup machine")
    add_parser.add_argument("-n", "--count", type=int, help="Number of virtual machines per cloud", default=1)

    add_parser.add_argument("-r", "--random-sleep", type=int, help="Randomise sleep before starting the process", default=int(config["TUNING"]["random_sleep_vm_create"]))
    add_parser.add_argument("-T", "--setup-timeout", type=int, help="Timeout for setup", default=int(config["TUNING"]["timeout_vm_setup"]))
    add_parser.add_argument("-A", "--max-attempts", type=int, help="Maximum number of attempts to setup the virtual machine", default=int(config["TUNING"]["max_it_vm_setup"]))
    add_parser.add_argument("-S", "--sleep", type=int, help="Sleep time between attempts to setup the virtual machine", default=int(config["TUNING"]["sleep_vm_setup"]))
    add_parser.add_argument("-at", "--add-timeout", type=int, help="Timeout for creating the virtual machine", default=int(config["TUNING"]["timeout_vm_create"]))
    add_parser.add_argument("-a", "--max-add-attempts", type=int, help="Maximum number of attempts to add the virtual machine", default=int(config["TUNING"]["max_it_vm_create"]))
    add_parser.add_argument("-s", "--add-sleep-time", type=int, help="Sleep time between attempts to add the virtual machine", default=int(config["TUNING"]["wait_vm_create"]))
    add_parser.add_argument("-vv", "--verbose-setup", action="store_true", help="Print verbose output for setup scripts", default=False)
    add_parser.add_argument("-e", "--heuristic", type=str, help="Heuristic to use for selecting the flavor", default="cpu-ram-disk")

    # create subparser for rm
    rm_parser = subparsers.add_parser("rm", help="Delete the virtual machines")

    rm_parser.add_argument("-m", "--metadata", metavar="KEY=VALUE", nargs='+', help="The metadata of the virtual machines to remove", default=None)
    rm_parser.add_argument("-c", "--clouds", nargs="+", help="Clouds to remove the virtual machine from", default=[])
    rm_parser.add_argument("-s", "--sleep", type=int, help="Sleep at the end of the process", default=int(config["TUNING"]["wait_after_delete"]))
    rm_parser.add_argument("-a", "--max-attempts", type=int, help="Maximum number of attempts to remove the virtual machine", default=int(config["TUNING"]["max_it_vm_delete"]))
    rm_parser.add_argument("-t", "--tokens", metavar="KEY=VALUE", nargs="+", help="Tokens to apply to the machines")

    # create subparser for exec
    exec_parser = subparsers.add_parser("exec", help="Execute a script on virtual machines")
    
    group = exec_parser.add_mutually_exclusive_group()
    group.add_argument("-S", "--script", help="Script to execute")
    group.add_argument("-C", "--commands", nargs="+",help="Commands to execute")
    group.add_argument("-F", "--file", help="File to execute")
    group.required = True
    
    exec_parser.add_argument("-t", "--tokens", metavar="KEY=VALUE", nargs="+", help="Tokens to apply the script", default=None)
    exec_parser.add_argument("-m", "--metadata", metavar="KEY=VALUE", nargs='+', help="The metadata of the virtual machines to  execute the script on", default=None)
    exec_parser.add_argument("-c", "--clouds", nargs="+", help="Clouds to execute the script on", default=[])
    exec_parser.add_argument("-H", "--hide", action="store_true", help="Hide the output of the script", default=False)
    exec_parser.add_argument("-T", "--timeout", type=int, help="Timeout for the script", default=int(config["TUNING"]["script_connect_timeout"]))
    exec_parser.add_argument("-a", "--max-attempts", type=int, help="Maximum number of attempts to execute the script", default=int(config["TUNING"]["max_it_vm_execute_script"]))
    exec_parser.add_argument("-s", "--sleep", type=int, help="Sleep time between attempts to execute the script", default=int(config["TUNING"]["wait_vm_execute_script"]))
    exec_parser.add_argument("-ca", "--max-cmd-attempts", type=int, help="Maximum number of attempts to execute the command", default=int(config["TUNING"]["max_it_vm_execute_cmd"]))
    exec_parser.add_argument("-cs", "--sleep-cmd", type=int, help="Sleep time between attempts to execute the command", default=int(config["TUNING"]["wait_vm_execute_cmd"]))
    exec_parser.add_argument("-n", "--count", type=int, help="Number of virtual machines to consider", default=float("inf"))
    exec_parser.add_argument("-e", "--heuristic", type=str, help="How to sort the machines", default="enum")


    # create subparser for update
    update_parser = subparsers.add_parser("update", help="Update the virtual machines")
    update_parser.add_argument("specs", metavar="KEY=VALUE", nargs='+', help="The metadata to update")

    update_parser.add_argument("-m", "--metadata", metavar="KEY=VALUE", nargs='+', help="The metadata of the virtual machines to update", default=None)
    update_parser.add_argument("-c", "--clouds", nargs="+", help="Clouds to remove the virtual machine from", default=[])
    update_parser.add_argument("-s", "--sleep", type=int, help="Sleep at the end of the process", default=int(config["TUNING"]["wait_after_update"]))
    update_parser.add_argument("-a", "--max-update-attempts", type=int, help="Maximum number of attempts to update the virtual machine", default=int(config["TUNING"]["max_it_vm_update"]))
    update_parser.add_argument("-t", "--tokens", metavar="KEY=VALUE", nargs="+", help="Tokens to apply to the machines")

    # create subparser for run
    run_parser = subparsers.add_parser("run", help="Run an execution pipeline")
    run_parser.add_argument("-f", "--file", help="File containing the execution pipeline", default="pipeline.json")
    run_parser.add_argument("-vv", "--verbose-setup", action="store_true", help="Print verbose output for setup scripts", default=False)

    # parse arguments
    args = parser.parse_args()

    # set verbose
    VERBOSE = args.verbose
    if VERBOSE:
        print("Verbose mode")

    if args.command == "setup":
        if VERBOSE:
            print("Setting up the clouds")
        setup_clouds(args.clouds, args.ips, verbose=args.verbose)
        if VERBOSE:
            print("Clouds setup")
    elif args.command == "status":
        if VERBOSE:
            print("Getting status of the clouds")
        args.metadata = parse_list_to_dict(args.metadata)
        status_clouds(args.clouds, args.metadata, verbose=args.verbose)
        if VERBOSE:
            print("Clouds status")
    elif args.command == "add":
        args.metadata = parse_list_to_dict(args.metadata)
        args.metadata = parse_dict_to_one_to_one_dict(args.metadata)
        args.tokens = parse_list_to_dict(args.tokens)
        args.tokens = parse_dict_to_one_to_one_dict(args.tokens)
        specs = {args.name: {"type":args.type, "clouds":{}, "metadata": args.metadata}}
        for cloud in args.clouds:
            specs[args.name]["clouds"][cloud] = args.count
        if VERBOSE:
            print("Adding the virtual machines")
        print(add_machines(specs, args.tokens, args.heuristic, args.random_sleep, args.setup_timeout, args.max_attempts, args.sleep, args.max_add_attempts, args.add_timeout, args.add_sleep_time, verbose=args.verbose, verbose_script=args.verbose_setup))
    elif args.command == "rm":
        sure = False
        args.metadata = parse_list_to_dict(args.metadata)
        args.tokens = parse_list_to_dict(args.tokens)
        args.tokens = parse_dict_to_one_to_one_dict(args.tokens)
        if args.metadata == {}:
            sure = input("Are you sure you want to remove all the virtual machines? (y/n) ") 
            sure = (sure.lower() == "y" or sure.lower() == "yes")
        if VERBOSE:
            print("Removing the virtual machines")
        print(remove_all_machines(args.metadata, args.clouds, args.tokens, args.sleep, args.max_attempts, verbose=args.verbose, sure=sure))
    elif args.command == "exec": 
        if VERBOSE:
            print("Executing the script on the virtual machines")
        args.metadata = parse_list_to_dict(args.metadata)
        args.tokens = parse_list_to_dict(args.tokens)
        args.tokens = parse_dict_to_one_to_one_dict(args.tokens)
        if args.script is not None:
            print(exec_script_machines(args.script, args.metadata, args.clouds, args.tokens, args.count, args.heuristic, args.hide, args.timeout, args.max_attempts, args.sleep, args.max_cmd_attempts, args.sleep_cmd, verbose=args.verbose))
        elif args.commands is not None:
            print(exec_script_machines(args.commands, args.metadata, args.clouds, args.tokens, args.count, args.heuristic, args.hide, args.timeout, args.max_attempts, args.sleep, args.max_cmd_attempts, args.sleep_cmd, verbose=args.verbose))
        elif args.file is not None:
            with open(args.file, "r") as f:
                script = f.readlines()
                for i in range(len(script)):
                    script[i] = script[i].strip().replace("\n", "")
            print(exec_script_machines(script, args.metadata, args.clouds, args.tokens, args.count, args.heuristic, args.hide, args.timeout, args.max_attempts, args.sleep, args.max_cmd_attempts, args.sleep_cmd, verbose=args.verbose))
    elif args.command == "update":
        if VERBOSE:
            print("Updating the virtual machines")
        args.specs = parse_list_to_dict(args.specs)
        args.specs = parse_dict_to_one_to_one_dict(args.specs)
        args.metadata = parse_list_to_dict(args.metadata)
        args.tokens = parse_list_to_dict(args.tokens)
        args.tokens = parse_dict_to_one_to_one_dict(args.tokens)
        print(update_machines(args.specs, args.metadata, args.clouds, args.tokens, args.sleep, args.max_update_attempts, verbose=args.verbose))
    elif args.command == "run":
        if VERBOSE:
            print("Running the execution pipeline")
        if not args.file.endswith(".json"):
            args.file += ".json"
        try:
            with open(args.file, "r") as f:
                pipeline = json.load(f)
            print(run(pipeline, verbose=args.verbose, verbose_script=args.verbose_setup))
        except Exception as e:
            print("Error: {}".format(e))
    if VERBOSE:
        print("Done")

import time
import json
import random
import pymongo
import argparse
import requests
import configparser

from datetime import datetime

import machines
import tasks

DIV = "_"
DIV1 = "-"
DIV2 = "/"

MANAGER_ID = None

CONFIG_FILE = "./config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
cw_config = config["CLOUDWATCHER"]

VERBOSE = True
VERY_VERBOSE = False
VERBOSE = True if VERY_VERBOSE else VERBOSE

client = pymongo.MongoClient("MONGODB_URL")
DB = client.cloudWatcher
COLLECTION = DB.report

def debug_print(*args, **kwargs):
    if kwargs.get("verbose", None) is None:
        kwargs["verbose"] = VERBOSE

    if MANAGER_ID is None:
        machines.debug_print("* cloudWatcher -", *args, **kwargs)
    else:
        machines.debug_print("* cloudWatcher > {} -".format(MANAGER_ID), *args, **kwargs)

def parse_results(results):
    parsed = results
    if results is not None:
        if type(results) is list:
            parsed = []
            for e in results:
                parsed.append(parse_results(e))
        elif type(results) is dict:
            parsed = {}
            for k,v in results.items():
                parsed[k] = parse_results(v)
        else:
            try:
                parsed = results.to_dict()
            except:
                parsed = results
            
    return parsed

def set_configuration(config_file=CONFIG_FILE, *, verbose=VERBOSE):
    global config
    global cw_config
    debug_print("Setting configuration file to {}".format(config_file), verbose=verbose)
    config.read(config_file)
    cw_config = config["CLOUDWATCHER"]

def reset_by_manager(manager_id, clouds=json.loads(cw_config["clouds"]), *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    debug_print("Resetting cloudWatcher manager {} network".format(manager_id), verbose=verbose)

    metadata = {
        "application":"cloudwatcher",
        "cloudwatcher-type":"probe",
        "cloudwatcher-manager":manager_id
    }

    report = machines.remove_all_machines(metadata, clouds, verbose=very_verbose)

    debug_print("Machines removed", verbose=verbose)
    return report

def reset(clouds=json.loads(cw_config["clouds"]), *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    debug_print("Resetting cloudWatcher network", verbose=verbose)

    metadata = {
        "application":"cloudwatcher"
    }

    report = machines.remove_all_machines(metadata, clouds, verbose=very_verbose)

    debug_print("Machines removed", verbose=verbose)
    return report

def setup_managers(manager_type="cw-manager", clouds=json.loads(cw_config["clouds"]), count=int(cw_config["managers"]), *, verbose=VERBOSE, very_verbose=VERY_VERBOSE, name=None):
    debug_print("Setting up cloudWatcher managers", verbose=verbose)
    debug_print("Creating {} managers for each cloud in {}".format(count, clouds), verbose=verbose)

    if name is None:
        name = "cloudWatcher"+DIV1+"manager"+DIV+"<CLOUD>"+DIV+"<INC_<CLOUD>>"

    specs = {
        "clouds": {
        },
        "metadata": {
            "application":"cloudwatcher",
            "cloudwatcher-type":"manager",
        },
        "type": manager_type
    }

    threads = []
    for cloud in clouds:
        specs["clouds"] = {cloud: count}
        named_specs = {name: specs}
        t = machines.ThreadWithReturnValue(target=machines.add_machines, args=(named_specs,), kwargs={"verbose": very_verbose, "verbose_script": very_verbose})
        threads.append((cloud,t))
        t.start()

    reports = {}
    for c,t in threads:
        if c not in reports:
            reports[c] = {}
        if manager_type not in reports[c]:
            reports[c][manager_type] = []
        reports[c][manager_type].append(t.join())
        

    reports = parse_results(reports)

    debug_print("Managers created", verbose=verbose)
    return reports

def setup(manager_id, clouds=json.loads(cw_config["clouds"]), count=int(cw_config["count"]), types=json.loads(cw_config["types"]), *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    debug_print("Setting up cloudWatcher manager {}".format(manager_id), verbose=verbose)
    debug_print("Creating {} machines per type in {} for each cloud in {}".format(count, types, clouds), verbose=verbose)

    name = "cloudWatcher"+DIV1+"probe"+DIV+((manager_id.replace("cloudWatcher"+DIV1+"manager"+DIV,"")).replace(DIV,DIV2))+DIV+"<INC_<CLOUD>>"+DIV+"<CLOUD>"

    specs = {
        "clouds": {
        },
        "metadata": {
            "application":"cloudwatcher",
            "cloudwatcher-type":"probe",
            "cloudwatcher-manager":manager_id
        }
    }

    threads = []
    for cloud in clouds:
        specs["clouds"] = {cloud: count}
        for type in types:
            typed_name = name+DIV+type
            named_specs = {typed_name: specs}
            named_specs[typed_name]["type"] = type
            t = machines.ThreadWithReturnValue(target=machines.add_machines, args=(named_specs,), kwargs={"verbose": verbose, "verbose_script": False})
            threads.append((cloud,type,t))
            t.start()

    reports = {}
    for c,type,t in threads:
        if c not in reports:
            reports[c] = {}
        if type not in reports[c]:
            reports[c][type] = []
        reports[c][type].append(t.join())
        

    reports = parse_results(reports)

    debug_print("Machines created", verbose=verbose)
    return reports

def start_manager(manager, config_file="./config.ini", clouds=json.loads(cw_config["clouds"]), managers=int(cw_config["managers"]), delta=float(cw_config["delta"]), verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    start_time = time.time()

    manager_name = manager.name
    metadata = manager.metadata

    while True:

        debug_print("Starting manager {}".format(manager_name), verbose=verbose)

        v_option = "-vv" if very_verbose else ("-v" if verbose else "-nv")

        time.sleep(60*random.random())

        machines.exec_script_machines("start", {"name": manager_name}, clouds, {"<CONFIG-FILE>": config_file,
                                                                                "<VERBOSE-OPTION>": v_option,
                                                                                "<RESET-OPTION>": "-r",
                                                                                "<MANAGER>": manager_name}, timeout_cmd=0, hide=False, verbose=very_verbose)

        debug_print("Manager {} stopped, delta {}, start_exec {}, time {}, diff {}".format(manager.name, delta, start_time, time.time(), time.time()-start_time), verbose=verbose)

        if delta != float("inf") and time.time()-start_time > delta:
            debug_print("Manager {} finished".format(manager.name), verbose=verbose)
            break
        else:
            while True:
                try:
                    debug_print("+++ Manager {} restarting".format(manager.name), verbose=verbose)
                    reset_by_manager(manager_name, clouds, verbose=verbose, very_verbose=very_verbose)
                    debug_print("--- Manager {}'s probes removed".format(manager.name), verbose=verbose)
                    print(machines.remove_all_machines({"name": manager_name}, [metadata["cloud"]], verbose=very_verbose))
                    debug_print("--- Manager {} removed".format(manager.name), verbose=verbose)
                    setup_managers(manager_type=metadata["type"], clouds=[metadata["cloud"]], count=1, verbose=verbose, very_verbose=very_verbose, name=manager_name)
                    debug_print("--- Manager {} recreated".format(manager.name), verbose=verbose)
                    break
                except Exception as e:
                    debug_print("Error while restarting manager {}: {}".format(manager.name, e), verbose=verbose)
                    time.sleep(60)



def start(config_file="./config.ini", clouds=json.loads(cw_config["clouds"]), count=int(cw_config["count"]), delta=float(cw_config["delta"]), verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    debug_print("Starting cloudWatcher's managers", verbose=verbose)
    managers = machines.get_servers({"application":"cloudwatcher","cloudwatcher-type":"manager"}, clouds, verbose=very_verbose)

    threads = []
    for manager in managers:
        t = machines.ThreadWithReturnValue(target=start_manager, args=(manager,config_file,clouds,count,delta), kwargs={"verbose": verbose, "very_verbose": very_verbose})
        threads.append(t)
        t.start()

    debug_print("All managers started", verbose=verbose)
    for t in threads:
        t.join()


def perform_task(task, machine, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    try:
        t = machines.ThreadWithReturnValue(target=getattr(tasks, task["function"]["name"]), args=(machine,task["args"]), kwargs={"verbose": verbose, "very_verbose": very_verbose})
        t.start()
        return t
    except Exception as e:
        print("Error performing task {} on machine {}: {}".format(task["function"]["name"], machine.name, e))
        return {}

def probe(machine, *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    results = {}
    debug_print("Probing machine {}".format(machine.name), verbose=very_verbose)

    results["started"] = datetime.now().isoformat()

    tasks = machines.get_tasks(machine.name)

    if tasks is not None:
        debug_print("Tasks found for machine {}: {}".format(machine.name, tasks), verbose=very_verbose)
        i = 1
        for step in tasks:
            debug_print("Running step {} for machine {}".format(i,machine.name), verbose=very_verbose)
            i+=1
            threads = []
            for task in step:
                if "args" not in task:
                    task["args"] = []
                t = perform_task(task, machine, verbose=verbose, very_verbose=very_verbose)
                threads.append((t,task["name"]))
            
            for t,task in threads:
                results[task] = t.join()
    else:
        debug_print("Error: No tasks for machine {}".format(machine.name), verbose=verbose)

    results["finished"] = datetime.now().isoformat()

    results["machine"] = machine.name

    results = parse_results(results)

    return results

def publish_report(report, *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    debug_print("Publishing report", verbose=verbose)

    try:
        requests.post("http://DASHBOARD_URL/report", json=report)
    except Exception as e:
        debug_print("Error publishing report: {}".format(e))

    try:
        COLLECTION.insert_one(report)
    except Exception as e:
        debug_print("Error inserting report: {}".format(e))

    time.sleep(1)

def check_slo(slo, data, *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    violations = {}
    if slo is not None and slo != {} and data is not None and data != {}:
        for k,v in slo.items():
            if k in data:
                if type(data[k]) is dict:
                    check = check_slo(v, data[k], verbose=verbose, very_verbose=very_verbose)
                    if check is not None and check != {}:
                        violations[k] = check
                else:
                    if "max" in v:
                        levels = v["max"]
                        if type(v["max"]) is not list:
                            levels = [[v["max"], "CRITICAL"]]

                        levels.sort(key=lambda x: x[0], reverse=True)

                        for level,severity in levels:
                            if data[k] > level:
                                violations[k] = {"severity": severity, "value": data[k], "max": level}

                    if "min" in v:
                        levels = v["min"]
                        if type(v["min"]) is not list:
                            levels = [(v["min"], "CRITICAL")]

                        levels.sort(key=lambda x: x[0])

                        for level,severity in levels:
                            if data[k] < level:
                                violations[k] = {"severity": severity, "value": data[k], "min": level}
    return violations

def process_results(manager_id, results, *, verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    report = {"reports": {}, "slo-violations": {}, "raw": results, "timestamp": datetime.now().isoformat(), "origin": manager_id}
    for cloud,types in results.items():
        for type,reports in types.items():
            for task in reports:
                if cloud not in report["reports"]:
                    report["reports"][cloud] = {}
                if type not in report["reports"][cloud]:
                    report["reports"][cloud][type] = {}
                machine = task["machine"]
                for task_name,task_report in task.items():
                    if task_name not in ["machine","started","finished"]:
                        if task_name not in report["reports"][cloud][type]:
                            report["reports"][cloud][type][task_name] = []
                        if task_report is not None and task_report != {}:
                            report["reports"][cloud][type][task_name].append(task_report)
                            task_data = machines.get_task(type=type, name=task_name)
                            if task_data is not None and "slo" in task_data["function"]:
                                violations = check_slo(task_data["function"]["slo"], task_report, verbose=verbose, very_verbose=very_verbose)
                                if violations != {}:
                                    if cloud not in report["slo-violations"]:
                                        report["slo-violations"][cloud] = {}
                                    if type not in report["slo-violations"][cloud]:
                                        report["slo-violations"][cloud][type] = {}
                                    if task_name not in report["slo-violations"][cloud][type]:
                                        report["slo-violations"][cloud][type][task_name] = []
                                    violations["source"] = machine
                                    report["slo-violations"][cloud][type][task_name].append(violations)
    
    for cloud,types in report["reports"].items():
        for type,task_reports in types.items():
            for task_name,task_report in task_reports.items():
                if task_name not in ["machine","started","finished"]:
                    task = machines.get_task(type=type, name=task_name)
                    if task is not None and task_report is not None and task_report != []:
                        report["reports"][cloud][type][task_name] = getattr(tasks, task["aggregate"]["name"])(task_report, verbose=verbose, very_verbose=very_verbose)
                        if "slo" in task["aggregate"]:
                            violations = check_slo(task["aggregate"]["slo"], report["reports"][cloud][type][task_name], verbose=verbose, very_verbose=very_verbose)
                            if violations != {}:
                                if cloud not in report["slo-violations"]:
                                    report["slo-violations"][cloud] = {}
                                if type not in report["slo-violations"][cloud]:
                                    report["slo-violations"][cloud][type] = {}
                                if task_name not in report["slo-violations"][cloud][type]:
                                    report["slo-violations"][cloud][type][task_name] = []
                                violations["source"] = "aggregate"
                                report["slo-violations"][cloud][type][task_name].append(violations)



    publish_report(report, verbose=verbose, very_verbose=very_verbose)
    return report


def run(manager_id, clouds=json.loads(cw_config["clouds"]), delta=float(cw_config["delta"]), types=json.loads(cw_config["types"]), count=int(cw_config["count"]), max_it=int(cw_config["max_it"]), wait=int(cw_config["wait"]), wait_it=int(cw_config["wait_it"]), *,  verbose=VERBOSE, very_verbose=VERY_VERBOSE):
    debug_print("Running cloudWatcher manager {} for {} seconds".format(manager_id, delta), verbose=verbose)
    start_time = time.time()
    debug_print("Running probes of manager {} for {} seconds".format(manager_id, delta), verbose=verbose)
    while True:

        for i in range(max_it):
            debug_print("Iteration {} of {}".format(i+1, max_it), verbose=verbose)
            machines_list = machines.get_servers({"application":"cloudwatcher", "cloudwatcher-type":"probe", "cloudwatcher-manager":manager_id}, clouds, verbose=very_verbose)

            if len(machines_list) < count*len(clouds)*len(types):
                debug_print("+++ Not enough machines to run the tests. Resetting", verbose=verbose)
                reset_by_manager(manager_id, clouds, verbose=verbose, very_verbose=very_verbose)
                setup(manager_id, clouds, count, types, verbose=verbose, very_verbose=very_verbose)

            threads = []
            for machine in machines_list:
                t = machines.ThreadWithReturnValue(target=probe, args=(machine,), kwargs={"verbose": verbose, "very_verbose": very_verbose})
                threads.append((t, machine.metadata["cloud"], machine.metadata["type"]))
                t.start()

            results = {}
            for t,cloud,type in threads:
                val = t.join()
                if cloud not in results:
                    results[cloud] = {}
                if type not in results[cloud]:
                    results[cloud][type] = []
                results[cloud][type].append(val)

            results = parse_results(results)

            process_results(manager_id, results, verbose=verbose)
        
            time.sleep(wait_it)

        time.sleep(wait)

        if delta != float("inf"):
            if time.time() - start_time > delta:
                break

    debug_print("cloudWatcher manager {} finished".format(manager_id), verbose=verbose)


if __name__ == "__main__":
    clouds = json.loads(cw_config["clouds"])
    count = int(cw_config["count"])
    managers = int(cw_config["managers"])
    types = json.loads(cw_config["types"])
    delta = float(cw_config["delta"])
    max_it = int(cw_config["max_it"])
    wait = int(cw_config["wait"])
    wait_it = int(cw_config["wait_it"])
    
    parser = argparse.ArgumentParser(description="CloudWatcher")

    parser.add_argument("-c", "--config", help="Configuration file", default=CONFIG_FILE)
    parser.add_argument("-nv", "--not-verbose", help="Verbose mode", action="store_true")
    parser.add_argument("-v", "--verbose", help="Verbose mode", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="Very verbose mode", action="store_true")
    parser.add_argument("-r", "--reset", help="Reset", action="store_true")

    # subparsers for commands
    subparsers = parser.add_subparsers(help="Subcommands", dest="command")

    # subparser for manager
    parser_manager = subparsers.add_parser("manager", help="Run a cloudWatcher manager")
    parser_manager.add_argument("-id", "--manager-id", help="Manager ID", default="cloudWatcher-manager")

    args = parser.parse_args()

    if args.not_verbose:
        VERBOSE = False
    if args.very_verbose:
        VERY_VERBOSE = True
        VERBOSE = True
    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        debug_print("Verbose mode", verbose=True)
    else:
        debug_print("Not verbose mode", verbose=True)
    if VERY_VERBOSE:
        debug_print("Very verbose mode", verbose=True)

    set_configuration(args.config, verbose=VERBOSE)

    if args.command == "manager":

        MANAGER_ID = args.manager_id

        if args.reset:
            reset_by_manager(MANAGER_ID, clouds=clouds, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

        setup(args.manager_id, clouds, count, types, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

        run(args.manager_id, clouds, delta, types, count, max_it, wait, wait_it, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

    else:
        debug_print("cloudWatcher starter", verbose=VERBOSE)
        if args.reset:
            reset(clouds=clouds, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

        start_time = time.time()
        while True:

            machines.setup_clouds(clouds, (count*len(types)*len(clouds)*managers)+managers, verbose=VERY_VERBOSE)

            debug_print("Setup managers", verbose=VERBOSE)
            setup_managers(clouds=clouds, count=managers, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

            debug_print("Start managers", verbose=VERBOSE)
            start(args.config, clouds=clouds, count=managers, delta=delta, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

            debug_print("cloudWatcher reset", verbose=VERBOSE)
            reset(clouds=clouds, verbose=VERBOSE, very_verbose=VERY_VERBOSE)

            if delta != float("inf"):
                if time.time() - start_time > delta:
                    break
        
        debug_print("cloudWatcher finisched after {} (delta: {})".format(time.time()-start_time,delta), verbose=VERBOSE)
    
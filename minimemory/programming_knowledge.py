"""Compact, license-friendly programming terminology seed data.

Definitions are original short explanations intended for local retrieval.
"""
from __future__ import annotations

TERMS = [
    # languages
    ("Python","language",None,"A general-purpose programming language.","Python is used to build software, automation, APIs, AI tools, and data applications."),
    ("JavaScript","language","JavaScript","A programming language commonly used for web and server applications.","JavaScript adds behavior to web pages and can also run on servers."),
    ("TypeScript","language","TypeScript","A typed language that extends JavaScript with static type syntax.","TypeScript helps describe and check the shapes of JavaScript data and code."),
    ("Kotlin","language","Kotlin","A modern statically typed language widely used for Android and JVM applications.","Kotlin is commonly used to build Android apps and JVM software."),
    ("Java","language","Java","A statically typed general-purpose language running on the JVM.","Java is used for applications, services, and many enterprise systems."),
    ("C","language","C","A compiled systems programming language with direct memory control.","C is often used for operating systems, embedded software, and low-level libraries."),
    ("C++","language","C++","A compiled language that extends C with higher-level programming features.","C++ is used for performance-sensitive applications, engines, and systems software."),
    ("Rust","language","Rust","A compiled systems language designed around performance and memory safety.","Rust aims to provide fast programs while preventing many memory-safety errors."),
    ("Go","language","Go","A compiled language designed for simplicity, concurrency, and network services.","Go is often used for APIs, command-line tools, and network services."),
    ("HTML","technology","HTML","A markup language for describing the structure of web documents.","HTML creates the structure and content of a webpage."),
    ("CSS","technology","CSS","A stylesheet language for controlling presentation of web documents.","CSS controls how a webpage looks and is laid out."),
    ("SQL","language","SQL","A language used to query and manipulate relational databases.","SQL is used to store, find, change, and organize data in relational databases."),
    # concepts
    ("variable","concept",None,"A named reference used to hold a value or object.","A variable gives a value a name so code can use it later."),
    ("constant","concept",None,"A named value intended not to be reassigned.","A constant represents a value that should stay fixed."),
    ("data type","concept",None,"A classification describing what kind of value data represents.","A data type tells code what kind of data a value contains."),
    ("function","concept",None,"A reusable unit of code that accepts inputs and may produce an output.","A function is reusable code that performs a task."),
    ("class","concept",None,"A definition used to create objects with data and behavior.","A class is a blueprint for objects."),
    ("object","concept",None,"A runtime value that combines state and behavior according to a model.","An object is a piece of data that can also provide actions."),
    ("method","concept",None,"A function associated with an object or class.","A method is a function that belongs to an object or class."),
    ("module","concept",None,"A file or package unit that provides reusable code and names.","A module is a reusable piece of a program."),
    ("package","concept",None,"A distributable collection of software modules.","A package groups code so it can be installed and reused."),
    ("library","concept",None,"Reusable software code intended to be called by another program.","A library gives your program ready-made functionality."),
    ("framework","concept",None,"A software structure that provides conventions and control flow for applications.","A framework provides a foundation and rules for building an application."),
    ("API","concept",None,"A defined interface through which software components communicate.","An API is a set of rules that lets one program use another program's functionality."),
    ("database","concept",None,"A system for storing and retrieving structured or semi-structured information.","A database stores information so software can find and update it."),
    ("algorithm","concept",None,"A defined procedure for solving a problem or transforming data.","An algorithm is a step-by-step method for solving a problem."),
    ("data structure","concept",None,"A representation used to organize and access data efficiently.","A data structure is a way to organize data for useful operations."),
    ("array","concept",None,"An indexed collection of values.","An array stores multiple values that can be accessed by position."),
    ("list","concept",None,"An ordered collection of values.","A list stores values in an ordered sequence."),
    ("dictionary","concept",None,"A mapping from keys to values.","A dictionary stores values that can be found by keys."),
    ("set","concept",None,"A collection designed to contain unique values.","A set stores unique values without requiring duplicates."),
    ("loop","concept",None,"A control structure that repeats a block of code.","A loop repeats code while a rule says it should continue."),
    ("condition","concept",None,"A boolean expression used to choose program behavior.","A condition lets a program make a decision."),
    ("exception","concept",None,"A runtime event representing an error or unusual condition.","An exception is a problem or special event that code can handle."),
    ("recursion","concept",None,"A technique in which a function calls itself on a smaller or simpler case.","Recursion solves a problem by repeatedly applying a function to smaller cases."),
    ("thread","concept",None,"A unit of execution within a process that can run concurrently with other threads.","A thread lets part of a program execute independently within a process."),
    ("process","concept",None,"A running instance of a program with its own operating-system resources.","A process is a program that is currently running."),
    ("concurrency","concept",None,"The ability to manage multiple tasks whose execution overlaps in time.","Concurrency lets software work on multiple tasks during the same period."),
    ("serialization","concept",None,"Converting data into a representation suitable for storage or transmission.","Serialization turns data into a storable or transferable format."),
    ("JSON","format",None,"A text format for representing structured data.","JSON is a common way to represent objects, lists, numbers, and text."),
    ("regex","concept",None,"A pattern language for matching and manipulating text.","A regular expression describes text patterns that code can search for."),
    # keywords
    ("if","keyword","Python","A conditional statement that runs code when an expression is true.","Use if to make a decision."),
    ("elif","keyword","Python","A conditional branch checked when earlier conditions were false.","Use elif to test another condition."),
    ("else","keyword",None,"A fallback branch used when preceding conditions are not selected.","Use else for the remaining case."),
    ("for","keyword",None,"A loop construct used to iterate over items or ranges.","Use for to repeat code for each item."),
    ("while","keyword",None,"A loop construct that repeats while a condition remains true.","Use while to repeat code while a condition is true."),
    ("return","keyword",None,"A statement that exits a function and can provide a result.","return sends a result back from a function."),
    ("import","keyword",None,"A statement or construct used to load code from a module or package.","import brings reusable code into a program."),
    ("class","keyword","Python","A Python statement used to define a class.","class starts a class definition in Python."),
    ("def","keyword","Python","A Python statement used to define a function.","def starts a function definition in Python."),
    ("try","keyword",None,"A construct used to run code that may raise an exception.","try starts code where an error may be handled."),
    ("except","keyword",None,"A construct used to handle selected exceptions.","except defines what to do when a matching error occurs."),
    ("async","keyword",None,"A keyword used to declare asynchronous functions in languages that support async programming.","async marks code that can work with asynchronous operations."),
    ("await","keyword",None,"A keyword used to suspend an asynchronous operation until an awaited result is ready.","await waits for an asynchronous result without blocking the intended async flow."),
    # web
    ("DOM","web_concept","JavaScript","The document object model represents a document as an object tree.","The DOM lets JavaScript inspect and change a webpage's structure."),
    ("HTTP","protocol",None,"An application protocol used for communication between clients and servers on the web.","HTTP defines how web clients and servers exchange requests and responses."),
    ("REST","architecture","HTTP","A style of designing network APIs around resources and standard HTTP operations.","REST is a common style for building web APIs."),
    ("request","web_concept","HTTP","A message sent by a client asking a server to perform an operation or return data.","A request asks a server for something or asks it to do something."),
    ("response","web_concept","HTTP","A message returned by a server after processing a request.","A response is what a server sends back to a client."),
    ("endpoint","web_concept","HTTP","A network-accessible API operation identified by a URL or route.","An endpoint is a specific address where an API provides an operation."),
    ("frontend","architecture",None,"The user-facing part of an application.","Frontend code is the part users interact with."),
    ("backend","architecture",None,"The server-side or non-user-facing application logic.","Backend code handles application logic, data, and services behind the interface."),
    ("full stack","architecture",None,"An application scope that includes both frontend and backend systems.","Full stack development covers the user interface and the backend."),
    # storage/devops
    ("SQLite","database","SQL","A small embedded relational database engine stored in a local file.","SQLite is a database that can run inside an application without a separate server."),
    ("migration","database_concept",None,"A controlled change to a database schema or stored-data structure.","A migration updates a database structure in a repeatable way."),
    ("cache","system_concept",None,"Temporary stored data used to speed up later access.","A cache keeps reusable results so they can be accessed faster."),
    ("dependency","software_concept",None,"Software required by another software component.","A dependency is code your program needs in order to work."),
    ("version","software_concept",None,"An identifier representing a particular release or state of software.","A version tells you which release of software you are using."),
    ("repository","software_concept",None,"A managed collection of source code and project history.","A repository stores a project's code and its development history."),
    ("compiler","tool","C++","A program that translates source code into another executable or intermediate representation.","A compiler translates source code into a form a computer can execute."),
    ("interpreter","tool",None,"A program that executes or evaluates source code through a runtime system.","An interpreter runs program instructions through a runtime."),
    ("debugger","tool",None,"A tool for inspecting and controlling program execution to locate errors.","A debugger helps you find and understand program errors."),
    ("unit test","testing",None,"A test that checks a small unit of program behavior in isolation.","A unit test checks one small part of a program."),
    ("integration test","testing",None,"A test that checks interactions between multiple software components.","An integration test checks whether connected parts work together."),
    ("static analysis","testing",None,"Analysis of source code without executing the program.","Static analysis checks code for likely problems without running it."),
    ("syntax","concept",None,"The rules governing the valid structure of source code.","Syntax is the grammar rules a programming language expects."),
    ("semantic","concept",None,"The meaning or behavior represented by code or data.","Semantics describe what code means or does."),
]


def seed(memory) -> int:
    """Insert or update the built-in programming terminology."""
    count = 0
    for term, category, language, meaning, simple in TERMS:
        memory.add_programming_term(term, category=category, language=language, meaning=meaning,
                                    simple_meaning=simple, source="minimemory-seed", license="Original short definitions; MIT project")
        count += 1
    return count

//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/program.hpp
// Author:            Martin Kirst
// Creation Date:     2011.11.16
// Description:
//
// Declaration of the main program instance.
// The main program instance is responsible for creating the window, managing the runtime behavior,
// controlling the main program sequence, reserving the render context
// and running the main execution loop.
//
////////////////////////////////////////////////////////////////////////////////

// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////
#include <SFML/Window.hpp>

#ifndef PROGRAM_HPP_
#define PROGRAM_HPP_

// Class Declaration
////////////////////////////////////////////////////////////////////////////////
class Program {
public:
	// Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////
	Program();
	~Program();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	bool IsRunning() const;
	unsigned int GetElapsedTime() const;
	unsigned long long GetTotalTime() const;
	float GetFramerate() const;

	// Public Methods
	////////////////////////////////////////////////////////////////////////////////
	void Run();
	void Exit();

private:
	// Private Methods
	////////////////////////////////////////////////////////////////////////////////
	void Initialize();
	void Update();
	void Draw();
	void HandleBasicEvents();

	// Variables
	////////////////////////////////////////////////////////////////////////////////
	sf::Window mainWindow;
	unsigned int timeCurrent;
	unsigned int timeLast;
	unsigned long long timeTotal;
};

#endif // PROGRAM_HPP_

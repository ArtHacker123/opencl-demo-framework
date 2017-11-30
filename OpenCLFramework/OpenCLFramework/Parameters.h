#pragma once
#define _CRTDBG_MAP_ALLOC
#ifdef _DEBUG
#define MYDEBUG_NEW   new( _NORMAL_BLOCK, __FILE__, __LINE__)
#define new MYDEBUG_NEW
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
//allocations to be of _CLIENT_BLOCK type
#else
#define MYDEBUG_NEW
#endif // _DEBUG
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <memory>

using namespace std;

template <class T>
struct Parameter;
template <class T>
std::ostream& operator<< (std::ostream& stream, const Parameter<T>& param);
std::ostream& operator<< (std::ostream& stream, const Parameter<bool>& param);

template <class T>
struct Parameter
{
	std::string name;
	std::string key;
	T value;

	Parameter(string nam, T val, string k) : name(nam), value(val), key(k) { cout << "init:" << name << endl; }

	~Parameter(){ cout << "dest:" << name
		<< endl; }
	friend std::ostream& operator<< (std::ostream& stream, const Parameter<T>& param);
	friend std::ostream& operator<< (std::ostream& stream, const Parameter<bool>& param);

	
};

template<typename T>
std::ostream& operator<< (std::ostream& stream, const Parameter<T>& param)
{
	stream << param.name << "(" << param.key << ")" << "=" << param.value;
	return stream;
}



class Parameters
{
	std::vector<std::shared_ptr<Parameter<bool>>> bparams;
	std::vector<std::shared_ptr<Parameter<int>>> iparams;
	std::vector<std::shared_ptr<Parameter<float>>> fparams;
public:
	Parameters();

	~Parameters();

	/*template <typename U>
	U get(string name, string type)
	{
		if (type == "float") return get_float(name);
		if (type == "bool") return get_bool(name);
		if (type == "int") return get_int(name);
	}*/

	float get_float(string name) const;
	bool get_bool(string name) const;
	int get_int(string name) const;
	void rem_float(const string key);
	/*bool pop_bool(string name);
	int pop_int(string name);*/
	
	friend std::ostream& operator<< (std::ostream& stream, const Parameters& param)
	{
		for (auto &it : param.bparams)
		{
			stream << *it << std::endl;
		}
		for (auto &it : param.iparams)
		{
			stream << *it << std::endl;
		}
		for (auto &it : param.fparams)
		{
			stream << *it << std::endl;
		}
		return stream;
	}

	void push(std::shared_ptr<Parameter<bool>> par)
	{
		bparams.push_back(par);
	}
	void push(std::shared_ptr<Parameter<int>> par)
	{
		iparams.push_back(par);
	}
	void push(std::shared_ptr<Parameter<float>> par)
	{
		fparams.push_back(std::move(par));
	}

	void change(string key, string val);

	void clear();
	void complete_clear();
};

class ParameterChangedException : public std::exception
{
	ParameterChangedException() {}
	~ParameterChangedException() {}
};


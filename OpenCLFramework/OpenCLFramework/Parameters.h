#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>

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

	Parameter(string nam, T val, string k) : name(nam), value(val), key(k) {}


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
	std::vector<Parameter<bool>*> bparams;
	std::vector<Parameter<int>*> iparams;
	std::vector<Parameter<float>*> fparams;
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
	
	friend std::ostream& operator<< (std::ostream& stream, const Parameters& param)
	{
		for (auto it : param.bparams)
		{
			stream << *it << std::endl;
		}
		for (auto it : param.iparams)
		{
			stream << *it << std::endl;
		}
		for (auto it : param.fparams)
		{
			stream << *it << std::endl;
		}
		return stream;
	}

	void push(Parameter<bool> *par)
	{
		bparams.push_back(par);
	}
	void push(Parameter<int> *par)
	{
		iparams.push_back(par);
	}
	void push(Parameter<float> *par)
	{
		fparams.push_back(par);
	}

	void change(string key, string val);

	void clear();
};
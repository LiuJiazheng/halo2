use core::panic;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Chip, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Selector, TableColumn},
    poly::Rotation,
};
use proptest::prelude::Rng;
use std::marker::PhantomData;

const MEM_RANGE_BITS: usize = 4;
const VALUE_RANGE_BITS: usize = 4;

struct MemTableChip<F: Field> {
    config: MemTableConfig,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct MemTableConfig {
    // | addr | is_addr_identical_bit | id | is_access_type_init | is_access_type_last_write | value |
    advice: [Column<Advice>; 6],
    // | binary_range | memory_range | value_range |
    range: [TableColumn; 3],
    // Selector for region
    sel: Selector,
}

impl<F: Field> Chip<F> for MemTableChip<F> {
    type Config = MemTableConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: Field> MemTableChip<F> {
    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// In MemTableConfig, we will constrain the following:
    /// r(cur).address == r(next).address => r(cur).id <= r(next).id
    /// r(cur).address != r(next).address <=> r(cur).access_type == last_write
    /// r(cur).address != r(next).address <=> r(next).access_type == init
    /// r(cur).address != r(next).address <=> r(cur).address < r(next).address
    /// Semenatically, it is a sort table, primary key is address, secondary key is id
    /// When address is different, last entry from previous group of address must be last write, and first enrty of next group of address must be init
    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 6],
        range: [TableColumn; 3],
        sel: Selector,
    ) -> <Self as Chip<F>>::Config {
        for col in advice.iter() {
            meta.enable_equality(*col);
        }

        let (addr, addr_delta_inv, id, is_access_type_init, is_access_type_last_write, value) =
            if let [addr, addr_delta_inv, id, is_access_type_init, is_access_type_last_write, value] =
                (0..6).map(|i| advice[i]).collect::<Vec<Column<Advice>>>()[..]
            {
                (
                    addr,
                    addr_delta_inv,
                    id,
                    is_access_type_init,
                    is_access_type_last_write,
                    value,
                )
            } else {
                panic!("wrong match")
            };
        let (binary_range, memory_range, value_range) = if let [binary_range, memory_range, value_range] =
            (0..3).map(|i| range[i]).collect::<Vec<TableColumn>>()[..]
        {
            (binary_range, memory_range, value_range)
        } else {
            panic!("wrong match")
        };

        // Lookup Conditions:
        // 2. Col acc_init_bit must be {0,1}
        // 3. Col acc_last_write_bit must be {0,1}
        // 4. id must be in memory range (for now)
        // 5. value must be in value range
        // 6. addr must be in memory range
        // 7. If addr_bit == 1, r(cur).id <= r(next).id
        // 8. If addr_bit == 0, r(cur).addr < r(next).addr
        // Cannot do a federal query since the tuple (a,b,c...) is not in the lookup table (bianry_range, binary_range, memorary_range...)
        // Each one should be a separate query

        meta.lookup(
            "memory tabel range check 2: acc_init_bit must be {0,1}",
            |meta| {
                let s = meta.query_selector(sel);
                let acc_init_bit = meta.query_advice(is_access_type_init, Rotation::next());

                vec![(s * acc_init_bit, binary_range)]
            },
        );

        meta.lookup(
            "memory tabel range check 3: acc_last_write_bit must be {0,1}",
            |meta| {
                let s = meta.query_selector(sel);
                let acc_last_write_bit =
                    meta.query_advice(is_access_type_last_write, Rotation::cur());

                vec![(s * acc_last_write_bit, binary_range)]
            },
        );

        meta.lookup(
            "memory table range check 4: id must be in memory range",
            |meta| {
                let s = meta.query_selector(sel);
                let cur_id = meta.query_advice(id, Rotation::cur());

                vec![(s * cur_id, memory_range)]
            },
        );

        meta.lookup(
            "memory table range check 5: value must be in value range",
            |meta| {
                let s = meta.query_selector(sel);
                let value = meta.query_advice(value, Rotation::cur());

                vec![(s * value, value_range)]
            },
        );

        meta.lookup(
            "memory table range check 6: addr must be in memory range",
            |meta| {
                let s = meta.query_selector(sel);
                let cur_addr = meta.query_advice(addr, Rotation::cur());

                vec![(s * cur_addr, memory_range)]
            },
        );

        meta.lookup(
            "memory table range check 7: same address, id check",
            |meta| {
                // Query selector
                let s = meta.query_selector(sel);
                // Get the detla of id
                let cur_id = meta.query_advice(id, Rotation::cur());
                let next_id = meta.query_advice(id, Rotation::next());
                let delta_id = next_id - cur_id.clone();
                // Query delta of address
                let cur_addr = meta.query_advice(addr, Rotation::cur());
                let next_addr = meta.query_advice(addr, Rotation::next());
                let delta_addr = next_addr - cur_addr;
                // Query inverse of delta of address
                let delta_addr_inv = meta.query_advice(addr_delta_inv, Rotation::next());
                // Is zero bit
                let is_zero_bit =
                    Expression::Constant(F::ONE) - delta_addr.clone() * delta_addr_inv;

                vec![(s * is_zero_bit * delta_id, memory_range)]
            },
        );

        meta.lookup(
            "memory table range check 8: different address, address stricly ascent",
            |meta| {
                // Query selector
                let s = meta.query_selector(sel);
                // Get the delta between the current address and the next address
                let cur_addr = meta.query_advice(addr, Rotation::cur());
                let next_addr = meta.query_advice(addr, Rotation::next());
                let delta_addr = next_addr - cur_addr.clone();
                // Query inverse of delta of address
                let delta_addr_inv = meta.query_advice(addr_delta_inv, Rotation::next());
                // Prepare constant
                let one = Expression::Constant(F::ONE);
                // Is zero bit
                let is_zero_bit = one.clone() - delta_addr.clone() * delta_addr_inv;

                vec![(
                    s * (one.clone() - is_zero_bit) * (delta_addr - one),
                    memory_range,
                )]
            },
        );

        meta.create_gate("memory table gate constraint", |meta| {
            let s = meta.query_selector(sel);
            // Get the delta between the current address and the next address
            let cur_addr = meta.query_advice(addr, Rotation::cur());
            let next_addr = meta.query_advice(addr, Rotation::next());
            let delta_addr = next_addr - cur_addr;

            // Query inverse of delta of address
            let delta_addr_inv = meta.query_advice(addr_delta_inv, Rotation::next());
            // Prepare constant
            let one = Expression::Constant(F::ONE);
            // Is zero bit
            let is_zero_bit = one.clone() - delta_addr.clone() * delta_addr_inv;
            // Query bits
            let acc_init_bit = meta.query_advice(is_access_type_init, Rotation::next());
            let acc_last_write_bit = meta.query_advice(is_access_type_last_write, Rotation::cur());

            // Conditions:
            // 1. Col addr_bit must obey the senmatics of "is next and cur addr equal", that means, if addr_bit == 1, r(cur).addr - r(next).addr == 0
            //    and if addr_bit == 0, r(cur).addr - r(next).addr != 0, more precisely, r(cur).addr - r(next).addr < 0, this part has been checked in the lookup table
            //    By leveaving the inverse of delta_addr, we can make that is_zero_bit == 1 iff delta_addr == 0
            // 2. addresses are different, iff access type of next addr must be init, essentially addr_bit xor acc_init_bit = 1 always holds
            // 3. addresses are different, iff access type of cur addr must be last write, essentially addr_bit xor acc_last_write_bit = 1 always holds
            // Use predicate logic, say A := (addr_bit == 1) B := (acc_last_write_bit == 1), (NOT A -> B) AND  (B -> NOT A)
            // reduces to
            // (A OR B) AND (NOT A OR NOT B) == TRUE
            // (NOT A AND NOT B) OR (A AND B) == FALSE

            vec![
                s.clone() * is_zero_bit.clone() * delta_addr,
                s.clone()
                    * ((one.clone() - is_zero_bit.clone()) * (one.clone() - acc_init_bit.clone())
                        + is_zero_bit.clone() * acc_init_bit.clone()),
                s * ((one.clone() - is_zero_bit.clone())
                    * (one.clone() - acc_last_write_bit.clone())
                    + is_zero_bit * acc_last_write_bit),
            ]
        });

        MemTableConfig { advice, range, sel }
    }

    fn assign_range(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let config = &self.config;
        let (binary_range, memory_range, value_range) = if let [binary_range, memory_range, value_range] =
            (0..3)
                .map(|i| config.range[i])
                .collect::<Vec<TableColumn>>()[..]
        {
            (binary_range, memory_range, value_range)
        } else {
            panic!("wrong match")
        };

        let res_binary = layouter.assign_table(
            || "assign binary table",
            |mut table| {
                table.assign_cell(|| "binary range", binary_range, 0, || Value::known(F::ZERO))?;
                table.assign_cell(|| "bianry range", binary_range, 1, || Value::known(F::ONE))?;
                Ok(())
            },
        );

        let mut acc = F::ZERO;
        let res_mem = layouter.assign_table(
            || "assign memory table",
            |mut table| {
                for i in 0..(1 << MEM_RANGE_BITS) {
                    table.assign_cell(|| "memory range", memory_range, i, || Value::known(acc))?;
                    acc += F::ONE;
                }
                Ok(())
            },
        );

        acc = F::ZERO;
        let res_val = layouter.assign_table(
            || "assign value table",
            |mut table| {
                for i in 0..(1 << VALUE_RANGE_BITS) {
                    table.assign_cell(|| "value range", value_range, i, || Value::known(acc))?;
                    acc += F::ONE;
                }
                Ok(())
            },
        );

        // Make sure no res is in error
        let res = [res_binary, res_mem, res_val];
        for res in res.iter() {
            if let Err(_) = res {
                return Err(Error::Synthesis);
            }
        }
        Ok(())
    }

    fn assign_table(
        &self,
        mut layouter: impl Layouter<F>,
        entries: &[MemTableEntry<F>],
    ) -> Result<Vec<MemTableEntry<F>>, Error> {
        let config = &self.config;
        macro_rules! is_not_equal {
            ($lhs:expr, $rhs:expr) => {
                if $lhs != $rhs {
                    Value::known(F::ONE)
                } else {
                    Value::known(F::ZERO)
                }
            };
        }

        let (addr, addr_delta_inv, id, is_access_type_init, is_access_type_last_write, value) =
            if let [addr, addr_delta_inv, id, is_access_type_init, is_access_type_last_write, value] =
                (0..6)
                    .map(|i| config.advice[i])
                    .collect::<Vec<Column<Advice>>>()[..]
            {
                (
                    addr,
                    addr_delta_inv,
                    id,
                    is_access_type_init,
                    is_access_type_last_write,
                    value,
                )
            } else {
                panic!("wrong match")
            };
        
        // Prepare vec for return
        let mut lw_entries: Vec<MemTableEntry<F>> = vec![];

        // Allocate mem table entries
        let _ = layouter.assign_region(
            || "assign mem table",
            |mut region: Region<F>| {
                // | addr | is_addr_identical_bit | id | is_access_type_init | is_access_type_last_write | value | sel |
                for (i, entry) in entries.iter().enumerate() {
                    // First one, no need for addr_bit
                    if i == 0 {
                        region.assign_advice(
                            || "first addr",
                            addr,
                            i,
                            || Value::known(entry.addr),
                        )?;
                        region.assign_advice(|| "first id", id, i, || Value::known(entry.id))?;
                        region.assign_advice(
                            || "first init",
                            is_access_type_init,
                            i,
                            || Value::known(F::ONE),
                        )?;
                        if entries.len() > 1 {
                            region.assign_advice(
                                || "first write be last write",
                                is_access_type_last_write,
                                i,
                                || is_not_equal!(entry.addr, entries[i + 1].addr),
                            )?;

                            if entry.addr != entries[i + 1].addr {
                                // store last write entry
                                lw_entries.push(MemTableEntry {
                                    addr: entry.addr,
                                    id: entry.id,
                                    value: entry.value,
                                });
                            }
                        }
                        region.assign_advice(
                            || "first value",
                            value,
                            i,
                            || Value::known(entry.value),
                        )?;
                        config.sel.enable(&mut region, i)?;
                    }
                    // Last one, no need for sel
                    else if i == entries.len() - 1 {
                        region.assign_advice(
                            || "last addr",
                            addr,
                            i,
                            || Value::known(entry.addr),
                        )?;
                        if entries.len() > 1 {
                            region.assign_advice(
                                || "last addr_bit",
                                addr_delta_inv,
                                i,
                                || {
                                    Value::known(
                                        (entry.addr - entries[i - 1].addr)
                                            .invert()
                                            .unwrap_or(F::ZERO),
                                    )
                                },
                            )?;
                        }
                        region.assign_advice(|| "last id", id, i, || Value::known(entry.id))?;
                        region.assign_advice(
                            || "last init",
                            is_access_type_init,
                            i,
                            || is_not_equal!(entry.addr, entries[i - 1].addr),
                        )?;
                        region.assign_advice(
                            || "last write",
                            is_access_type_last_write,
                            i,
                            || Value::known(F::ONE),
                        )?; // Must be last write for sure
                        region.assign_advice(
                            || "last value",
                            value,
                            i,
                            || Value::known(entry.value),
                        )?;
                        // Must be last write entry
                        lw_entries.push(MemTableEntry {
                            addr: entry.addr,
                            id: entry.id,
                            value: entry.value,
                        });
                    }
                    // Other rows
                    else {
                        region.assign_advice(
                            || format!("{} addr", i),
                            addr,
                            i,
                            || Value::known(entry.addr),
                        )?;
                        region.assign_advice(
                            || format!("{} addr_bit", i),
                            addr_delta_inv,
                            i,
                            || {
                                Value::known(
                                    (entry.addr - entries[i - 1].addr)
                                        .invert()
                                        .unwrap_or(F::ZERO),
                                )
                            },
                        )?;
                        region.assign_advice(
                            || format!("{} id", i),
                            id,
                            i,
                            || Value::known(entry.id),
                        )?;
                        region.assign_advice(
                            || format!("{} init", i),
                            is_access_type_init,
                            i,
                            || is_not_equal!(entry.addr, entries[i - 1].addr),
                        )?;
                        region.assign_advice(
                            || format!("{} write", i),
                            is_access_type_last_write,
                            i,
                            || is_not_equal!(entry.addr, entries[i + 1].addr),
                        )?;
                        region.assign_advice(
                            || format!("{} value", i),
                            value,
                            i,
                            || Value::known(entry.value),
                        )?;
                        config.sel.enable(&mut region, i)?;
                        // If it is last write, store it
                        if entry.addr != entries[i + 1].addr {
                            lw_entries.push(MemTableEntry {
                                addr: entry.addr,
                                id: entry.id,
                                value: entry.value,
                            });
                        }
                    }
                }
                Ok(())
            },
        )?;

        Ok(lw_entries)
    }
}

struct LastWriteTableChip<F: Field> {
    config: LastWriteTableConfig,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct LastWriteTableConfig {
    // | addr | id | value | heritage |
    advice: [Column<Advice>; 4],
    // selector for region
    sel: Selector,
}

impl<F: Field> Chip<F> for LastWriteTableChip<F> {
    type Config = LastWriteTableConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: Field> LastWriteTableChip<F> {
    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Configure the last write table
    /// memtbl_schema: [addr, id, value, is_last_write]
    /// last_write_schema: [addr, id, value, heritage]
    fn configure(
        meta: &mut ConstraintSystem<F>,
        memtbl_schema: [Column<Advice>; 4],
        memtbl_sel: Selector,
        last_write_schema: [Column<Advice>; 4],
        last_write_sel: Selector,
        binary_range: TableColumn,
    ) -> <Self as Chip<F>>::Config {
        // Enforce equality
        for col in memtbl_schema.iter() {
            meta.enable_equality(*col);
        }

        for col in last_write_schema.iter() {
            meta.enable_equality(*col);
        }

        // Load name of columns
        let (memtbl_addr, memtbl_id, memtbl_value, is_last_write) = if let [addr, id, value, is_last_write] =
            (0..4).map(|i| memtbl_schema[i]).collect::<Vec<Column<Advice>>>()[..]
        {
            (addr, id, value, is_last_write)
        } else {
            panic!("wrong match")
        };

        let (lw_addr, lw_id, lw_value, heritage) = if let [addr, id, value, heritage] =
            (0..4).map(|i| last_write_schema[i]).collect::<Vec<Column<Advice>>>()[..]
        {
            (addr, id, value, heritage)
        } else {
            panic!("wrong match")
        };

        // Lookup Conditions:
        // heritage must be in binary range
        meta.lookup("lw table col heritage", |meta| {
            let s = meta.query_selector(last_write_sel);
            let heritage = meta.query_advice(heritage, Rotation::cur());

            vec![(s * heritage, binary_range)]
        });

        // Lookup Conditions:
        // is_last_write must be in binary range
        meta.lookup("memtbl col is_last_write", |meta| {
            let s = meta.query_selector(memtbl_sel);
            let is_last_write = meta.query_advice(is_last_write, Rotation::cur());

            vec![(s * is_last_write, binary_range)]
        });

        // Lookup Conditions:
        // For any row in lw table, it comes from a row in mem table
        meta.lookup_any("lw tbl belongs to memtbl", |meta| {
            let s = meta.query_selector(last_write_sel);
            let lw_addr = meta.query_advice(lw_addr, Rotation::cur());
            let lw_id = meta.query_advice(lw_id, Rotation::cur());
            let lw_value = meta.query_advice(lw_value, Rotation::cur());

            let memtbl_addr = meta.query_advice(memtbl_addr, Rotation::cur());
            let memtbl_id = meta.query_advice(memtbl_id, Rotation::cur());
            let memtbl_value = meta.query_advice(memtbl_value, Rotation::cur());

            vec![
                (s.clone() * lw_addr, memtbl_addr),
                (s.clone() * lw_id, memtbl_id),
                (s * lw_value, memtbl_value),
            ]
        });

        // Lookup Conditions:
        // For any row in mem table, and it is marked as 'last write', it must belong to a row in lw table
        // Thus, lw table is exact the extraction of mem table with 'last write' marked -- every last write entry from memtbl has
        // been present in lw table, and only once. The uniqueness of lw table entry is guaranteed by the uniqueness of last write entry of mem table
        meta.lookup_any("memtbl last write entry belongs to lw table", |meta| {
            let s = meta.query_selector(memtbl_sel);
            let memtbl_addr = meta.query_advice(memtbl_addr, Rotation::cur());
            let memtbl_id = meta.query_advice(memtbl_id, Rotation::cur());
            let memtbl_value = meta.query_advice(memtbl_value, Rotation::cur());
            let is_last_write = meta.query_advice(is_last_write, Rotation::cur());

            let lw_addr = meta.query_advice(lw_addr, Rotation::cur());
            let lw_id = meta.query_advice(lw_id, Rotation::cur());
            let lw_value = meta.query_advice(lw_value, Rotation::cur());

            vec![
                (s.clone() * memtbl_addr * is_last_write.clone(), lw_addr),
                (s.clone() * memtbl_id * is_last_write.clone(), lw_id),
                (s * memtbl_value * is_last_write, lw_value),
            ]
        });

        LastWriteTableConfig {
            advice: last_write_schema,
            sel: last_write_sel,
        }
    }

    #[allow(unused)]
    fn zero_padding_one_row(
        mut layouter: impl Layouter<F>,
        schema: [Column<Advice>; 3],
        sel: Selector,
    ) -> Result<(), Error> {
        let (addr, id, value) = if let [addr, id, value] =
            (0..3).map(|i| schema[i]).collect::<Vec<Column<Advice>>>()[..]
        {
            (addr, id, value)
        } else {
            panic!("wrong match")
        };

        layouter.assign_region(
            || "zero padding one row",
            |mut region: Region<F>| {
                region.assign_advice(|| "addr", addr, 0, || Value::known(F::ZERO))?;
                region.assign_advice(|| "id", id, 0, || Value::known(F::ZERO))?;
                region.assign_advice(|| "value", value, 0, || Value::known(F::ZERO))?;
                sel.enable(&mut region, 0)?;
                Ok(())
            },
        )   
    }

    fn assign_lwtbl_from_memtbl(
        &self,
        mut layouter: impl Layouter<F>,
        memtbl_entries: &[MemTableEntry<F>],
    ) -> Result<(), Error> {
        let config = &self.config;
        let (lw_addr, lw_id, lw_value, heritage) = if let [addr, id, value, heritage] =
            (0..4).map(|i| config.advice[i]).collect::<Vec<Column<Advice>>>()[..]
        {
            (addr, id, value, heritage)
        } else {
            panic!("wrong match")
        };

        // Allocate lwtbl based on the given entries
        layouter.assign_region(
            || "assign lw table",
            |mut region: Region<F>| {
                for (i, entry) in memtbl_entries.iter().enumerate() {
                    region.assign_advice(
                        || format!("{} addr", i),
                        lw_addr,
                        i,
                        || Value::known(entry.addr),
                    )?;
                    region.assign_advice(
                        || format!("{} id", i),
                        lw_id,
                        i,
                        || Value::known(entry.id),
                    )?;
                    region.assign_advice(
                        || format!("{} value", i),
                        lw_value,
                        i,
                        || Value::known(entry.value),
                    )?;
                    // This one is 'extracted' from memtbl, not a heritage
                    region.assign_advice(
                        || format!("{} heritage", i),
                        heritage,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    config.sel.enable(&mut region, i)?;
                }
                Ok(())
            },
        )
    }
}

#[derive(Clone, Debug)]
struct MemTableEntry<F: Field> {
    addr: F,
    id: F,
    value: F,
}

#[derive(Clone, Debug)]
struct CircuitConfig {
    memtbl_config: MemTableConfig,
    lwtbl_config: LastWriteTableConfig,
}

#[derive(Default, Debug)]
struct MinimalMemTable<F: Field> {
    entries: Vec<MemTableEntry<F>>,
}

impl<F> Circuit<F> for MinimalMemTable<F>
where
    F: Field,
{
    type Config = CircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let memtbl_advice = [(); 6].map(|_| meta.advice_column());
        let range = [(); 3].map(|_| meta.lookup_table_column());
        let memtbl_sel = meta.complex_selector();
    
        let lw_advice = [(); 4].map(|_| meta.advice_column());
        let lw_sel = meta.complex_selector();
        // Reference memtbl schema
        // | addr | id | value | is_last_write |
        // Order really matters, latter we can make it several const
        let ref_memtbl_schema = [memtbl_advice[0], memtbl_advice[2], memtbl_advice[5], memtbl_advice[4]];
        let binary_range = range[0];

        let memtbl_config = MemTableChip::configure(meta, memtbl_advice, range, memtbl_sel);
        let lwtbl_config = LastWriteTableChip::configure(
            meta,
            ref_memtbl_schema,
            memtbl_sel,
            lw_advice,
            lw_sel,
            binary_range,
        );

        CircuitConfig {
            memtbl_config,
            lwtbl_config,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Get MemTableChip
        let memtbl_chip = MemTableChip::<F>::construct(config.memtbl_config);
        // Assign range
        memtbl_chip.assign_range(layouter.namespace(|| "assign range"))?;
        // Assign table
        let lwtbl_from_memtbl = memtbl_chip.assign_table(layouter.namespace(|| "assign table"), &self.entries).unwrap();

        for entry in lwtbl_from_memtbl.iter() {
            println!("addr: {:?}, id: {:?}, value: {:?}", entry.addr, entry.id, entry.value);
        }
        let lwtbl_from_memtbl = lwtbl_from_memtbl[..lwtbl_from_memtbl.len() - 1].to_vec();
        // Get LastWriteTableChip
        let lwtbl_chip = LastWriteTableChip::<F>::construct(config.lwtbl_config);
        // Memtbl schema | addr | id | value |
        let ref_memtbl_schema = [
            memtbl_chip.config().advice[0],
            memtbl_chip.config().advice[2],
            memtbl_chip.config().advice[5],
        ];
        // Lwtbk schema | addr | id | value |
        let ref_lwtbl_schema = [
            lwtbl_chip.config().advice[0],
            lwtbl_chip.config().advice[1],
            lwtbl_chip.config().advice[2],
        ];
        // Assign zero padding on memtbl
        //LastWriteTableChip::<F>::zero_padding_one_row(layouter.namespace(|| "zero padding on memtbl"), ref_memtbl_schema, memtbl_chip.config.sel)?;
        // Assign zero padding on lwtbl
        //LastWriteTableChip::<F>::zero_padding_one_row(layouter.namespace(|| "zero padding on lwtbl"), ref_lwtbl_schema, lwtbl_chip.config.sel)?;

        // Assign lwtbl from memtbl
        lwtbl_chip.assign_lwtbl_from_memtbl(layouter.namespace(|| "assign lwtbl from memtbl"), &lwtbl_from_memtbl)?;

        Ok(())
    }
}

fn main() {
    use std::{
        fs::File,
        io::{BufReader, BufWriter, Write},
    };

    use halo2_proofs::dev::MockProver;
    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::SingleStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    };
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand_core::OsRng;

    // ANCHOR: test-circuit
    // The number of rows in our circuit cannot exceed 2^k. Since our example
    // circuit is very small, we can pick a very small value here.
    let k = 8;

    // Create an buffer
    let mut entries: Vec<MemTableEntry<Fr>> = vec![];

    // Prepare the private inputs to the circuit!
    let mut rng = OsRng;
    for id in 0..(1 << MEM_RANGE_BITS) {
        // we only genegate 6 addresses, by Pigeonhole principle there must be some addresses with more than one entry
        let addr = Fr::from(rng.gen_range(0..6) as u64);
        let value = Fr::from(rng.gen_range(0..(1 << VALUE_RANGE_BITS)) as u64);
        entries.push(MemTableEntry {
            addr,
            id: Fr::from(id as u64),
            value,
        });
    }

    // Sort the entries by address and then by id
    entries.sort_by(|a, b| {
        if a.addr == b.addr {
            a.id.cmp(&b.id)
        } else {
            a.addr.cmp(&b.addr)
        }
    });

    println!("Sorted Entries are: ");
    for entry in entries.iter() {
        println!("addr: {:?}, id: {:?}, value: {:?}", entry.addr, entry.id, entry.value);
    }
    println!("End of sorted entries");

    // Create the circuit
    let circuit = MinimalMemTable { entries };

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));

    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    let vk = keygen_vk(&params, &circuit).expect("vk should not fail");
    let pk = keygen_pk(&params, vk, &circuit).expect("pk should not fail");

    let f = File::create("serialization-test.pk").unwrap();
    let mut writer = BufWriter::new(f);
    pk.write(&mut writer, SerdeFormat::RawBytes).unwrap();
    writer.flush().unwrap();

    let f = File::open("serialization-test.pk").unwrap();
    let mut reader = BufReader::new(f);
    #[allow(clippy::unit_arg)]
    let pk = ProvingKey::<G1Affine>::read::<_, MinimalMemTable<Fr>>(
        &mut reader,
        SerdeFormat::RawBytes,
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    )
    .unwrap();

    std::fs::remove_file("serialization-test.pk").unwrap();

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverGWC<'_, Bn256>,
        Challenge255<G1Affine>,
        _,
        Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
        _,
    >(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
    .expect("prover should not fail");
    let proof = transcript.finalize();

    let strategy = SingleStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    assert!(verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierGWC<'_, Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<'_, Bn256>,
    >(&params, pk.get_vk(), strategy, &[&[]], &mut transcript)
    .is_ok());

    println!("Proof and verification succeed!");
}
